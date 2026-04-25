"""PyQt5 GUI for Neighbor2Neighbor Bayer denoising training.

Layout
------
+-------------------------------------------------------------+
|  [config inputs] [Start] [Stop]                             |
|  [progress bar] [elapsed / budget]                          |
+--------------------+----------------------------------------+
|                    |       64x64 patch preview              |
|   Loss curve       |   noisy (left) | denoised (right)      |
|   (log-log)        |   nearest-neighbour upscaled 16x       |
|                    |                                        |
+--------------------+----------------------------------------+
|  status / log                                               |
+-------------------------------------------------------------+

The training runs on a ``QThread``; the worker emits Qt signals carrying
``StepInfo`` / ``PreviewInfo`` payloads. After training the GUI offers a
"Generate result/" button to run inference on the full dataset.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from n2n.infer import run_inference_set
from n2n.raw_utils import (
    gray_world_gains,
    packed_to_display,
    raw_to_display,
    raw_to_linear_bgr,
    unpack_rggb,
)
from n2n.trainer import PreviewInfo, StepInfo, TrainConfig, Trainer

PREVIEW_MIN = 256  # smallest side of the preview label, in pixels


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------
class TrainWorker(QtCore.QObject):
    step_signal = QtCore.pyqtSignal(object)       # StepInfo
    preview_signal = QtCore.pyqtSignal(object)    # PreviewInfo
    finished_signal = QtCore.pyqtSignal(str)      # checkpoint path
    error_signal = QtCore.pyqtSignal(str)
    log_signal = QtCore.pyqtSignal(str)

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _on_step(self, info: StepInfo):
        self.step_signal.emit(info)

    def _on_preview(self, info: PreviewInfo):
        self.preview_signal.emit(info)

    def _on_finish(self, ckpt: Path):
        self.finished_signal.emit(str(ckpt))

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.log_signal.emit("Trainer starting...")
            trainer = Trainer(
                self.cfg,
                on_step=self._on_step,
                on_preview=self._on_preview,
                on_finish=self._on_finish,
                is_cancelled=lambda: self._cancelled,
            )
            trainer.run()
        except Exception:  # noqa: BLE001
            self.error_signal.emit(traceback.format_exc())


class InferWorker(QtCore.QObject):
    progress_signal = QtCore.pyqtSignal(int, int, str)
    finished_signal = QtCore.pyqtSignal()
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, ckpt_path: str, data_root: str, out_root: str):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.data_root = data_root
        self.out_root = out_root
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _cb(self, i: int, total: int, path: Path):
        self.progress_signal.emit(i, total, str(path))

    @QtCore.pyqtSlot()
    def run(self):
        try:
            run_inference_set(
                self.ckpt_path,
                self.data_root,
                self.out_root,
                progress_cb=self._cb,
                is_cancelled=lambda: self._cancelled,
            )
            self.finished_signal.emit()
        except Exception:  # noqa: BLE001
            self.error_signal.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# GUI widgets
# ---------------------------------------------------------------------------
class LossPlot(FigureCanvas):
    """Per-epoch loss plot with three log-log curves: main / reg / total.

    The denoising-quality metric is ``main`` (L1 between f-denoised g1 and
    g2) - it converges to the irreducible noise floor and is the right
    thing to monitor. ``reg`` is the consistency regulariser; it ramps
    from 0 as the model becomes non-trivial and is expected to grow.
    """

    def __init__(self, parent=None):
        fig = Figure(figsize=(4.5, 3.5))
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel("epoch (log)")
        self.ax.set_ylabel("mean L1 (log)")
        self.ax.set_title("per-epoch training loss (log-log)")
        self.ax.grid(True, which="both", linestyle=":", alpha=0.4)
        (self._main_line,) = self.ax.plot(
            [], [], color="#1f77b4", linewidth=1.6, marker="o", markersize=4, label="main L1"
        )
        (self._reg_line,) = self.ax.plot(
            [], [], color="#d62728", linewidth=1.0, marker="x", markersize=4, label="reg L1"
        )
        (self._total_line,) = self.ax.plot(
            [], [], color="#7f7f7f", linewidth=0.8, linestyle="--", label="total"
        )
        self.ax.legend(loc="upper right", fontsize=8)
        self._xs: list[int] = []
        self._main: list[float] = []
        self._reg: list[float] = []
        self._total: list[float] = []
        fig.tight_layout()

    def append(self, epoch: int, main: float, reg: float, total: float):
        if not np.isfinite(main) or not np.isfinite(total):
            return
        self._xs.append(epoch)
        self._main.append(max(main, 1e-9))
        self._reg.append(max(reg, 1e-9))
        self._total.append(max(total, 1e-9))
        self._main_line.set_data(self._xs, self._main)
        # Only show reg / total when reg is actually being optimised; otherwise
        # they coincide with main and clutter the plot.
        reg_active = max(self._reg) > 1e-7
        if reg_active:
            self._reg_line.set_data(self._xs, self._reg)
            self._total_line.set_data(self._xs, self._total)
        else:
            self._reg_line.set_data([], [])
            self._total_line.set_data([], [])
        self.ax.set_xlim(max(1, self._xs[0]), max(2, self._xs[-1] * 1.1))
        ys = self._main + (self._reg + self._total if reg_active else [])
        lo = max(min(ys) * 0.6, 1e-6)
        hi = max(ys) * 1.4
        self.ax.set_ylim(lo, hi)
        self.draw_idle()

    def reset(self):
        self._xs.clear()
        self._main.clear()
        self._reg.clear()
        self._total.clear()
        for ln in (self._main_line, self._reg_line, self._total_line):
            ln.set_data([], [])
        self.draw_idle()


class _ZoomLabel(QtWidgets.QLabel):
    """QLabel that re-renders its source image with NEAREST when resized."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(PREVIEW_MIN, PREVIEW_MIN)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.setStyleSheet("background:#202020;color:#888;border:1px solid #555")
        self.setText("(no preview yet)")
        self._src: np.ndarray | None = None

    def set_source(self, rgb: np.ndarray):
        """Cache source RGB image; trigger a NEAREST re-render at current size."""
        self._src = np.ascontiguousarray(rgb)
        self._redraw()

    def resizeEvent(self, event):  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._redraw()

    def _redraw(self):
        if self._src is None:
            return
        side = max(PREVIEW_MIN, min(self.width(), self.height()))
        big = cv2.resize(self._src, (side, side), interpolation=cv2.INTER_NEAREST)
        big = np.ascontiguousarray(big)
        h, w = big.shape[:2]
        qimg = QtGui.QImage(big.data, w, h, w * 3, QtGui.QImage.Format_RGB888).copy()
        self.setPixmap(QtGui.QPixmap.fromImage(qimg))


class PreviewPanel(QtWidgets.QWidget):
    """Two side-by-side image labels for noisy / denoised preview.

    Re-renders with INTER_NEAREST whenever the panel is resized, so the
    source 64x64 (4-ch packed -> 128x128 demosaiced) preview pixels remain
    crisp at any zoom level the user picks.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.noisy_box = self._make_box("noisy")
        self.denoised_box = self._make_box("denoised")

        layout.addWidget(self.noisy_box[0])
        layout.addWidget(self.denoised_box[0])

    def _make_box(self, title: str):
        box = QtWidgets.QGroupBox(title)
        box.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        v = QtWidgets.QVBoxLayout(box)
        lbl = _ZoomLabel()
        v.addWidget(lbl)
        return box, lbl

    def update_preview(self, info: PreviewInfo):
        # Share WB gains between noisy & denoised so the colour balance is
        # identical and the only visible difference is noise level. WB
        # gains are computed in 16-bit linear domain (no uint8 quantisation).
        bayer_den = unpack_rggb(info.denoised_packed)
        gains = gray_world_gains(raw_to_linear_bgr(bayer_den))

        noisy = raw_to_display(
            unpack_rggb(info.noisy_packed), wb_gains=gains, return_rgb=True
        )
        den = raw_to_display(bayer_den, wb_gains=gains, return_rgb=True)
        self.noisy_box[1].set_source(noisy)
        self.denoised_box[1].set_source(den)
        self.noisy_box[0].setTitle(f"noisy @ epoch {info.epoch}  (ISO=16, NN, WB)")
        self.denoised_box[0].setTitle(f"denoised @ epoch {info.epoch}  (ISO=16, NN, WB)")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *, default_seconds: int = 600, auto_result: bool = False):
        super().__init__()
        self.setWindowTitle("Neighbor2Neighbor 8-bit Bayer Denoiser - Trainer")
        self.resize(1280, 820)

        self._train_thread: QtCore.QThread | None = None
        self._train_worker: TrainWorker | None = None
        self._infer_thread: QtCore.QThread | None = None
        self._infer_worker: InferWorker | None = None

        self._budget_seconds: float = float(default_seconds)
        self._t0: float = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._auto_result = bool(auto_result)

        self._build_ui()
        self.train_secs_edit.setValue(int(default_seconds))

    # -- UI construction ---------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # --- top: config row ---
        top = QtWidgets.QHBoxLayout()

        self.data_root_edit = QtWidgets.QLineEdit("train_data/Data")
        self.train_secs_edit = QtWidgets.QSpinBox()
        self.train_secs_edit.setRange(5, 36000)
        self.train_secs_edit.setValue(600)
        self.train_secs_edit.setSuffix(" s")
        self.batch_edit = QtWidgets.QSpinBox()
        self.batch_edit.setRange(1, 32)
        self.batch_edit.setValue(6)
        self.patch_edit = QtWidgets.QSpinBox()
        self.patch_edit.setRange(32, 512)
        self.patch_edit.setSingleStep(32)
        self.patch_edit.setValue(256)
        self.lr_edit = QtWidgets.QDoubleSpinBox()
        self.lr_edit.setDecimals(5)
        self.lr_edit.setRange(1e-5, 1e-2)
        self.lr_edit.setSingleStep(1e-4)
        self.lr_edit.setValue(3e-4)
        self.fp16_cb = QtWidgets.QCheckBox("FP16")
        self.fp16_cb.setChecked(True)

        for label, widget in [
            ("data root", self.data_root_edit),
            ("train", self.train_secs_edit),
            ("batch", self.batch_edit),
            ("patch", self.patch_edit),
            ("lr", self.lr_edit),
        ]:
            top.addWidget(QtWidgets.QLabel(label))
            top.addWidget(widget)
        top.addWidget(self.fp16_cb)
        top.addStretch(1)

        self.start_btn = QtWidgets.QPushButton("Start training")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)

        outer.addLayout(top)

        # --- progress row ---
        prog = QtWidgets.QHBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.elapsed_label = QtWidgets.QLabel("00:00 / 00:00")
        self.step_label = QtWidgets.QLabel("step 0   loss 0.000000")
        prog.addWidget(self.progress_bar, 4)
        prog.addWidget(self.step_label, 3)
        prog.addWidget(self.elapsed_label, 2)
        outer.addLayout(prog)

        # --- middle: loss + preview ---
        middle = QtWidgets.QHBoxLayout()
        self.loss_plot = LossPlot()
        self.preview = PreviewPanel()
        middle.addWidget(self.loss_plot, 4)
        middle.addWidget(self.preview, 6)
        outer.addLayout(middle, 5)

        # --- bottom: log + result button ---
        bottom = QtWidgets.QHBoxLayout()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        bottom.addWidget(self.log, 4)

        right_box = QtWidgets.QVBoxLayout()
        self.status_label = QtWidgets.QLabel("idle")
        self.status_label.setStyleSheet("font-weight:bold;font-size:14px;color:#888")
        self.result_btn = QtWidgets.QPushButton("Generate result/ comparison")
        self.result_btn.setEnabled(False)
        self.result_btn.clicked.connect(self.on_generate_result)
        right_box.addWidget(self.status_label)
        right_box.addStretch(1)
        right_box.addWidget(self.result_btn)
        bottom.addLayout(right_box, 1)

        outer.addLayout(bottom, 1)

    # -- helpers -----------------------------------------------------------
    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    @staticmethod
    def _fmt(t: float) -> str:
        t = int(max(0, t))
        return f"{t // 60:02d}:{t % 60:02d}"

    def _tick(self):
        elapsed = time.time() - self._t0
        self.elapsed_label.setText(
            f"{self._fmt(elapsed)} / {self._fmt(self._budget_seconds)}"
        )
        progress = min(1.0, elapsed / max(1.0, self._budget_seconds))
        self.progress_bar.setValue(int(progress * 1000))

    # -- training control --------------------------------------------------
    def on_start(self):
        if self._train_thread is not None:
            return
        cfg = TrainConfig(
            data_root=self.data_root_edit.text().strip() or "train_data/Data",
            train_seconds=float(self.train_secs_edit.value()),
            batch_size=int(self.batch_edit.value()),
            patch_size=int(self.patch_edit.value()),
            lr=float(self.lr_edit.value()),
            use_fp16=bool(self.fp16_cb.isChecked()),
        )
        self.loss_plot.reset()
        self._budget_seconds = cfg.train_seconds
        self._t0 = time.time()
        self._timer.start()

        self.progress_bar.setValue(0)
        self.status_label.setText("training...")
        self.status_label.setStyleSheet("font-weight:bold;font-size:14px;color:#1c6")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.result_btn.setEnabled(False)

        self._train_thread = QtCore.QThread()
        self._train_worker = TrainWorker(cfg)
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.step_signal.connect(self.on_step)
        self._train_worker.preview_signal.connect(self.preview.update_preview)
        self._train_worker.finished_signal.connect(self.on_train_finished)
        self._train_worker.error_signal.connect(self.on_train_error)
        self._train_worker.log_signal.connect(self._log)
        self._train_thread.start()

    def on_stop(self):
        if self._train_worker is not None:
            self._train_worker.cancel()
            self._log("Stop requested; finishing current step...")
        if self._infer_worker is not None:
            self._infer_worker.cancel()

    @QtCore.pyqtSlot(object)
    def on_step(self, info: StepInfo):
        self.loss_plot.append(info.epoch, info.main_loss, info.reg_loss, info.loss)
        self.step_label.setText(
            f"epoch {info.epoch}  step {info.step}  "
            f"main {info.main_loss:.5f}  reg {info.reg_loss:.5f}  "
            f"total {info.loss:.5f}  ({info.steps_per_sec:.1f} it/s)"
        )

    @QtCore.pyqtSlot(str)
    def on_train_finished(self, ckpt_path: str):
        self._timer.stop()
        self.progress_bar.setValue(1000)
        self.status_label.setText("TRAINING FINISHED")
        self.status_label.setStyleSheet("font-weight:bold;font-size:16px;color:#0a0")
        self._log(f"Checkpoint saved to {ckpt_path}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.result_btn.setEnabled(True)
        self._ckpt_path = ckpt_path
        self._train_thread.quit()
        self._train_thread.wait()
        self._train_thread = None
        self._train_worker = None
        if self._auto_result:
            self._log("Auto-launching result generation...")
            QtCore.QTimer.singleShot(200, self.on_generate_result)

    @QtCore.pyqtSlot(str)
    def on_train_error(self, tb: str):
        self._timer.stop()
        self.status_label.setText("ERROR")
        self.status_label.setStyleSheet("font-weight:bold;font-size:14px;color:#c33")
        self._log(tb)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self._train_thread:
            self._train_thread.quit()
            self._train_thread.wait()
        self._train_thread = None
        self._train_worker = None

    # -- result generation -------------------------------------------------
    def on_generate_result(self):
        if self._infer_thread is not None:
            return
        ckpt = getattr(self, "_ckpt_path", None) or "checkpoints/n2n_model.pt"
        self.status_label.setText("inferring...")
        self.status_label.setStyleSheet("font-weight:bold;font-size:14px;color:#1c6")
        self.result_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._infer_thread = QtCore.QThread()
        self._infer_worker = InferWorker(
            ckpt_path=ckpt,
            data_root=self.data_root_edit.text().strip() or "train_data/Data",
            out_root="result",
        )
        self._infer_worker.moveToThread(self._infer_thread)
        self._infer_thread.started.connect(self._infer_worker.run)
        self._infer_worker.progress_signal.connect(self.on_infer_progress)
        self._infer_worker.finished_signal.connect(self.on_infer_finished)
        self._infer_worker.error_signal.connect(self.on_train_error)
        self._infer_thread.start()

    @QtCore.pyqtSlot(int, int, str)
    def on_infer_progress(self, i: int, total: int, path: str):
        self.progress_bar.setValue(int(1000 * i / max(1, total)))
        self.step_label.setText(f"infer {i}/{total}")
        self._log(f"  wrote {path}")

    @QtCore.pyqtSlot()
    def on_infer_finished(self):
        self.status_label.setText("RESULTS WRITTEN to result/")
        self.status_label.setStyleSheet("font-weight:bold;font-size:16px;color:#0a0")
        self.result_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._infer_thread.quit()
        self._infer_thread.wait()
        self._infer_thread = None
        self._infer_worker = None


def parse_args(argv):
    p = argparse.ArgumentParser(description="N2N denoising training GUI")
    p.add_argument("seconds", type=int, nargs="?", default=600,
                   help="training wall-clock budget in seconds (default 600 = 10 min)")
    p.add_argument("--auto-start", action="store_true",
                   help="press Start as soon as the window opens")
    p.add_argument("--auto-result", action="store_true",
                   help="automatically run inference & write `result/` after training ends")
    return p.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    app = QtWidgets.QApplication([sys.argv[0]])
    win = MainWindow(default_seconds=args.seconds, auto_result=args.auto_result)
    win.show()
    if args.auto_start:
        QtCore.QTimer.singleShot(300, win.on_start)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
