"""
EEG Info Panel
==============
Dual-mode info bar: static overview (batch) / live monitor (stream).
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame


class EEGInfoPanel(QFrame):
    """Info bar showing loaded EEG data.

    Batch mode — static summary after loading.
    Stream mode — live monitor simulating amplifier status bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "background-color: #2a2a2a; border-radius: 4px;"
        )
        self.setMaximumHeight(44)
        self.setVisible(False)
        self._mode: str = "batch"
        self._labels: dict[str, QLabel] = {}
        self._field_widgets: list[QWidget] = []

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(10, 4, 10, 4)
        self._layout.setSpacing(14)
        self._layout.addStretch()

    # ----------------------------------------------------------------
    # Field helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _key_style() -> str:
        return "color: #777; font-size: 11px;"

    @staticmethod
    def _val_style() -> str:
        return "color: #00cc66; font-size: 12px; font-weight: bold;"

    @staticmethod
    def _dim_style() -> str:
        return "color: #888; font-size: 12px;"

    def _add_field(self, key: str) -> QLabel:
        """Append `key : [value]` pair to the right of the layout."""
        k = QLabel(key)
        k.setStyleSheet(self._key_style())
        v = QLabel("—")
        v.setStyleSheet(self._val_style())
        # remove trailing stretch, insert widgets, re-add stretch
        self._layout.takeAt(self._layout.count() - 1)
        self._layout.addWidget(k)
        self._layout.addWidget(v)
        self._layout.addStretch()
        self._labels[key] = v
        self._field_widgets.extend([k, v])
        return v

    def _set_value(self, key: str, text: str, highlight: bool = True):
        if key in self._labels:
            self._labels[key].setText(text)
            self._labels[key].setStyleSheet(
                self._val_style() if highlight else self._dim_style()
            )

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def show_batch(self, source) -> None:
        """Populate with static overview (batch mode)."""
        self._clear_fields()
        self._mode = "batch"

        name = _display_name(source)
        n_ch = source.n_channels
        sfreq = source.sfreq
        duration = source.total_samples / sfreq if sfreq > 0 else 0

        ch_names = _channel_label(source)

        self._add_field("File")
        self._add_field("Channels")
        self._add_field("Rate")
        self._add_field("Duration")
        if ch_names:
            self._add_field("Names")

        self._set_value("File", name)
        self._set_value("Channels", f"{n_ch} ch")
        self._set_value("Rate", f"{sfreq:.0f} Hz")
        self._set_value("Duration", f"{duration:.1f}s")
        if ch_names:
            self._set_value("Names", ch_names, highlight=False)

        self.setVisible(True)

    def show_stream(self, source) -> None:
        """Set up live monitor layout (stream mode)."""
        self._clear_fields()
        self._mode = "stream"

        name = _display_name(source)
        n_ch = source.n_channels
        sfreq = source.sfreq
        total_sec = source.total_samples / sfreq if sfreq > 0 else 0

        self._add_field("Session")
        self._add_field("Rate")
        self._add_field("Elapsed")
        self._add_field("Impedance")

        self._set_value("Session", name)
        self._set_value("Rate", f"{sfreq:.0f} Hz / {n_ch} ch")
        self._set_value("Elapsed", f"00:00 / {_fmt_time(total_sec)}")
        self._set_value("Impedance", "✔ OK", highlight=True)

        self.setVisible(True)

    def update_elapsed(self, source) -> None:
        """Update live elapsed time in stream mode."""
        if self._mode != "stream":
            return
        elapsed = source.position / source.sfreq if source.sfreq > 0 else 0
        total = source.total_samples / source.sfreq if source.sfreq > 0 else 0
        self._set_value("Elapsed", f"{_fmt_time(elapsed)} / {_fmt_time(total)}")

    def set_filter_status(self, enabled: bool, l_freq: float, h_freq: float):
        """Show/hide and update filter indicator in stream mode."""
        if self._mode != "stream":
            return
        key = "Filter"
        if enabled:
            text = f"BP {l_freq:.1f}–{h_freq:.1f} Hz"
            if key not in self._labels:
                self._add_field(key)
            self._set_value(key, text, highlight=True)
        else:
            if key not in self._labels:
                self._add_field(key)
            self._set_value(key, "OFF", highlight=False)

    def clear(self) -> None:
        self._clear_fields()
        self.setVisible(False)

    def _clear_fields(self):
        for w in self._field_widgets:
            self._layout.removeWidget(w)
            w.deleteLater()
        self._field_widgets.clear()
        self._labels.clear()


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _display_name(source) -> str:
    import re
    stem = source.filepath.stem
    m = re.match(r'^(.*)R\d+$', stem)
    return m.group(1) if m else stem


def _channel_label(source) -> str:
    try:
        if source._raws:
            raw = source._raws[0]
            if hasattr(raw, 'ch_names'):
                names = raw.ch_names[:6]
                suffix = f" +{len(raw.ch_names) - 6}" if len(raw.ch_names) > 6 else ""
                return ", ".join(names) + suffix
    except Exception:
        pass
    return ""


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
