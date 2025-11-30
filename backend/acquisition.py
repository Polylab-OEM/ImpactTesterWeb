#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acquisition worker for LabJack U6/U6-Pro.
Python 3.9 compatible (no | types).
- Supports multiple analog channels.
- Optional analog or digital trigger with pre/post samples.
- Writes CSV + events + errors + history.
- Calls sample_callback with decimated scaled samples for live plotting.
"""

import os
import csv
import time
import math
import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Callable, Dict

try:
    import u6
    from u6 import U6
except Exception as e:  # pragma: no cover
    U6 = None
    _u6_import_error = e
else:
    _u6_import_error = None


# ---------------- Data classes (mirrored with backend.main) ----------------
@dataclass
class ChannelConfig:
    ain: int = 0
    label: str = "CH0"
    multiplier: float = 1.0
    enabled: bool = True


@dataclass
class TriggerConfig:
    enable: bool = True
    trigger_ain: int = 0
    threshold_volts: float = 1.0
    rising: bool = True

    use_digital: bool = False
    digital_line: int = 0
    digital_rising: bool = True
    digital_level_mode: bool = False
    digital_active_high: bool = True
    digital_debounce_ms: int = 0  # kept for compatibility; not used in this worker

    pre_samples: int = 4000
    post_samples: int = 4000
    safety_timeout_sec: float = 15.0


@dataclass
class StreamConfig:
    scan_frequency: int = 50000
    resolution_index: int = 0
    settling_factor: int = 0
    duration_sec: float = 5.0
    ui_plot_hz: int = 100


@dataclass
class AppSettings:
    client: str = ""
    item: str = ""
    notes: str = ""
    channels: List[ChannelConfig] = field(
        default_factory=lambda: [
            ChannelConfig(ain=0, label="Force", multiplier=1.0, enabled=True),
            ChannelConfig(ain=1, label="Aux", multiplier=1.0, enabled=False),
        ]
    )
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    output_dir: str = field(default_factory=lambda: os.path.abspath("."))
    export_only_after_trigger: bool = True


# ---------------- Helper ----------------
def dictify(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dictify(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [dictify(x) for x in obj]
    return obj


# ---------------- U6Streamer ----------------
class U6Streamer:
    """
    Runs in its own thread, controlled externally.
    Call .run() inside a thread target.
    """

    def __init__(
        self,
        settings: AppSettings,
        status_callback: Optional[Callable[[str], None]] = None,
        sample_callback: Optional[Callable[[list, list], None]] = None,
    ):
        self.settings = settings
        self.status_callback = status_callback or (lambda msg: None)
        self.sample_callback = sample_callback or (lambda t, y: None)

        self.stop_flag = False
        self._device = None

        self._csv = None
        self._csv_path = None
        self._events_csv = None
        self._events_writer = None
        self._errlog_csv = None
        self._err_writer = None
        self._history_csv = None
        self._history_writer = None

        self._sample_index = 0
        self._missed_total = 0

        self._triggered = False
        self._post_remaining = 0
        self._prebuf = []

        self._prev_trig_val = None
        self._prev_dig = None

    # ---- Public helpers ----
    @property
    def csv_path(self) -> Optional[str]:
        return self._csv_path

    def update_status(self, msg: str):
        self.status_callback(msg)

    # ---- DIO helpers ----
    def _dio_set_input(self, line: int) -> bool:
        try:
            self._device.getFeedback(u6.BitDirWrite(line, 0))  # 0 = input
            return True
        except Exception as e:
            self.update_status("Warning setting DIO input: %s" % e)
            return False

    def _dio_read(self, line: int) -> Optional[int]:
        try:
            rsp = self._device.getFeedback(u6.BitStateRead(line))
            if not rsp:
                return None
            return int(rsp[0])
        except Exception:
            return None

    # ---- Trigger helpers ----
    @staticmethod
    def _detect_cross(prev, curr, thr, rising=True) -> bool:
        if prev is None or curr is None:
            return False
        if rising:
            return prev < thr <= curr
        return prev > thr >= curr

    # ---- Device / file setup ----
    def _open_device(self, enabled_ch: List[int]):
        if U6 is None:
            raise RuntimeError("LabJack U6 library not available: %s" % _u6_import_error)
        self._device = U6()
        self._device.getCalibrationData()
        self.update_status("Connected to U6; calibration loaded.")

        trig = self.settings.trigger
        if trig.enable and trig.use_digital:
            line = int(trig.digital_line)
            self._dio_set_input(line)
            self._prev_dig = self._dio_read(line)
            self.update_status("DIO line %d INPUT; initial=%s" % (line, self._prev_dig))

        self._device.streamConfig(
            NumChannels=len(enabled_ch),
            ChannelNumbers=enabled_ch,
            ChannelOptions=[0] * len(enabled_ch),
            ResolutionIndex=self.settings.stream.resolution_index,
            SettlingFactor=self.settings.stream.settling_factor,
            ScanFrequency=self.settings.stream.scan_frequency,
        )

    def _open_outputs(self, enabled_ch: List[int]):
        out_dir = self.settings.output_dir or os.path.abspath(".")
        os.makedirs(out_dir, exist_ok=True)
        base = "u6log_%s" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(out_dir, base + ".csv")

        self._csv = open(self._csv_path, "w", newline="", encoding="utf-8")
        w = csv.writer(self._csv)

        header = ["index", "t_seconds"]
        for ch in self.settings.channels:
            if ch.enabled:
                header.append("AIN%d_V" % ch.ain)
                header.append("%s_scaled" % ch.label)
        w.writerow(header)

        self._events_csv = open(
            os.path.join(out_dir, base + "_events.csv"), "w", newline="", encoding="utf-8"
        )
        self._events_writer = csv.writer(self._events_csv)
        self._events_writer.writerow(
            [
                "event_time_iso",
                "sample_index",
                "t_seconds",
                "trigger_source",
                "trigger_ain",
                "digital_line",
                "threshold_volts",
                "rising",
                "csv_path",
            ]
        )

        self._errlog_csv = open(
            os.path.join(out_dir, base + "_errors.csv"), "w", newline="", encoding="utf-8"
        )
        self._err_writer = csv.writer(self._errlog_csv)
        self._err_writer.writerow(
            ["time_iso", "sample_index", "packet_errors", "missed", "note"]
        )

        hist_path = os.path.join(out_dir, "trigger_history.csv")
        init_hist = not os.path.exists(hist_path)
        self._history_csv = open(hist_path, "a", newline="", encoding="utf-8")
        self._history_writer = csv.writer(self._history_csv)
        if init_hist:
            self._history_writer.writerow(
                [
                    "event_time_iso",
                    "sample_index",
                    "t_seconds",
                    "trigger_source",
                    "trigger_ain",
                    "digital_line",
                    "threshold_volts",
                    "rising",
                    "csv_path",
                ]
            )

        return w

    # ---- Main run ----
    def run(self):
        try:
            enabled_cfg = [c for c in self.settings.channels if c.enabled]
            enabled_ch = [c.ain for c in enabled_cfg]
            if not enabled_ch:
                raise RuntimeError("No channels enabled.")

            trig = self.settings.trigger
            if trig.enable and (not trig.use_digital):
                if trig.trigger_ain not in enabled_ch:
                    raise RuntimeError(
                        "Trigger AIN must be one of the enabled channels (for analog trigger)."
                    )

            self._open_device(enabled_ch)
            writer = self._open_outputs(enabled_ch)

            dt = 1.0 / float(self.settings.stream.scan_frequency)
            start_time = time.perf_counter()

            self._sample_index = 0
            self._missed_total = 0
            self._triggered = False
            self._post_remaining = trig.post_samples
            self._prebuf = []
            self._prev_trig_val = None

            # UI decimation for sample_callback
            ui_plot_hz = max(10, int(self.settings.stream.ui_plot_hz))
            ui_decim = max(1, int(self.settings.stream.scan_frequency // ui_plot_hz))
            ui_counter = 0

            last_val = {ch.ain: 0.0 for ch in enabled_cfg}
            prev_val = {ch.ain: None for ch in enabled_cfg}
            SPIKE_DELTA_V = 5.0

            mode = "digital" if trig.use_digital else ("analog" if trig.enable else "none")
            self.update_status("Streaming... (trigger: %s)" % mode)

            # For batching samples to callback
            batch_t = []
            batch_scaled = []
            push_every = max(1, int(self.settings.stream.scan_frequency / 200.0))

            self._device.streamStart()

            for pkt in self._device.streamData():
                if pkt is None:
                    continue
                if self.stop_flag:
                    break

                if pkt.get("errors"):
                    err = pkt["errors"]
                    note = ""
                    if err == 48:
                        note = "Overrun (48). Lower sample rate / resolution / channels."
                    self._err_writer.writerow(
                        [
                            datetime.datetime.now().isoformat(),
                            self._sample_index,
                            err,
                            pkt.get("missed", 0),
                            note,
                        ]
                    )
                    self.update_status("Packet error: %s" % err)

                if pkt.get("missed"):
                    self._missed_total += pkt["missed"]

                n_samps = 0
                for ain in enabled_ch:
                    key = "AIN%d" % ain
                    if key in pkt and isinstance(pkt[key], list):
                        n_samps = max(n_samps, len(pkt[key]))
                if n_samps == 0:
                    continue

                packet_fired = False
                if trig.enable and trig.use_digital:
                    curr_dig = self._dio_read(trig.digital_line)
                    if trig.digital_level_mode:
                        active = 1 if trig.digital_active_high else 0
                        packet_fired = curr_dig == active
                    else:
                        if self._prev_dig is not None:
                            if trig.digital_rising:
                                packet_fired = (self._prev_dig == 0 and curr_dig == 1)
                            else:
                                packet_fired = (self._prev_dig == 1 and curr_dig == 0)
                    self._prev_dig = curr_dig

                for i in range(n_samps):
                    if self.stop_flag:
                        break

                    row_raw = []
                    row_scaled = []
                    trig_val = None

                    for ch in enabled_cfg:
                        vals = pkt.get("AIN%d" % ch.ain, [])
                        if i < len(vals):
                            v = float(vals[i])
                            last_val[ch.ain] = v
                        else:
                            v = last_val[ch.ain]

                        pv = prev_val[ch.ain]
                        if pv is not None and abs(v - pv) > SPIKE_DELTA_V:
                            v = pv
                        prev_val[ch.ain] = v

                        row_raw.append(v)
                        row_scaled.append(v * float(ch.multiplier))
                        if trig.enable and (not trig.use_digital) and ch.ain == trig.trigger_ain:
                            trig_val = v

                    t = self._sample_index * dt
                    fired = False

                    if not trig.enable:
                        pass
                    elif trig.use_digital:
                        fired = (not self._triggered) and packet_fired and (i == 0)
                    else:
                        if self._prev_trig_val is None:
                            self._prev_trig_val = trig_val
                        fired = self._detect_cross(
                            self._prev_trig_val,
                            trig_val,
                            trig.threshold_volts,
                            trig.rising,
                        )
                        self._prev_trig_val = trig_val

                    if not trig.enable:
                        if (ui_counter % ui_decim) == 0:
                            batch_t.append(t)
                            batch_scaled.append(list(row_scaled))
                            if len(batch_t) >= push_every:
                                self.sample_callback(batch_t, batch_scaled)
                                batch_t = []
                                batch_scaled = []
                        outrow = [self._sample_index, t]
                        for rv, sv in zip(row_raw, row_scaled):
                            outrow.extend([rv, sv])
                        writer.writerow(outrow)
                    else:
                        if not self._triggered:
                            outrow = [self._sample_index, t]
                            for rv, sv in zip(row_raw, row_scaled):
                                outrow.extend([rv, sv])
                            self._prebuf.append(outrow)

                            if trig.use_digital and trig.digital_level_mode:
                                if self._prev_dig is not None:
                                    active = 1 if trig.digital_active_high else 0
                                    if self._prev_dig == active:
                                        fired = True

                            if fired:
                                self._triggered = True
                                evt = datetime.datetime.now().isoformat()
                                src = "digital" if trig.use_digital else "analog"
                                dline = trig.digital_line if trig.use_digital else ""
                                row_evt = [
                                    evt,
                                    self._sample_index,
                                    t,
                                    src,
                                    trig.trigger_ain,
                                    dline,
                                    trig.threshold_volts,
                                    trig.rising,
                                    self._csv_path,
                                ]
                                self._events_writer.writerow(row_evt)
                                self._history_writer.writerow(row_evt)

                                if len(self._prebuf) > trig.pre_samples:
                                    self._prebuf = self._prebuf[-trig.pre_samples :]

                                for brow in self._prebuf:
                                    writer.writerow(brow)
                                self._prebuf = []
                                writer.writerow(outrow)
                                self._post_remaining = trig.post_samples
                            else:
                                if (ui_counter % ui_decim) == 0:
                                    batch_t.append(t)
                                    batch_scaled.append(list(row_scaled))
                                    if len(batch_t) >= push_every:
                                        self.sample_callback(batch_t, batch_scaled)
                                        batch_t = []
                                        batch_scaled = []

                                if (time.perf_counter() - start_time) > trig.safety_timeout_sec:
                                    self.update_status("No trigger before timeout, stopping.")
                                    for brow in self._prebuf:
                                        writer.writerow(brow)
                                    self._prebuf = []
                                    self.stop_flag = True
                                    break
                        else:
                            outrow = [self._sample_index, t]
                            for rv, sv in zip(row_raw, row_scaled):
                                outrow.extend([rv, sv])
                            writer.writerow(outrow)

                            if (ui_counter % ui_decim) == 0:
                                batch_t.append(t)
                                batch_scaled.append(list(row_scaled))
                                if len(batch_t) >= push_every:
                                    self.sample_callback(batch_t, batch_scaled)
                                    batch_t = []
                                    batch_scaled = []

                            self._post_remaining -= 1
                            if self._post_remaining <= 0:
                                self.update_status("Post-trigger complete, stopping.")
                                self.stop_flag = True
                                break

                    self._sample_index += 1
                    ui_counter += 1

                if self.stop_flag:
                    break

            if batch_t:
                self.sample_callback(batch_t, batch_scaled)

            if trig.enable and (not self._triggered) and self._prebuf:
                for brow in self._prebuf:
                    writer.writerow(brow)
                self._prebuf = []

            self.update_status(
                "Stopped. Samples=%d, missed=%d" % (self._sample_index, self._missed_total)
            )

        except Exception as e:  # pragma: no cover - runtime errors visible via status
            self.update_status("ERROR in acquisition: %s" % e)
        finally:
            try:
                if self._device is not None:
                    try:
                        self._device.streamStop()
                    except Exception:
                        pass
                    self._device.close()
            except Exception:
                pass

            for fh in (self._csv, self._events_csv, self._errlog_csv, self._history_csv):
                try:
                    if fh:
                        fh.flush()
                        fh.close()
                except Exception:
                    pass
