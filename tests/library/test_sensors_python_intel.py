import math
import unittest
from unittest.mock import patch

import library.sensors.sensors_python as sensors_python


class TestSensorsPythonIntelGpu(unittest.TestCase):
    def setUp(self):
        self.original_detected_gpu = sensors_python.DETECTED_GPU

    def tearDown(self):
        sensors_python.DETECTED_GPU = self.original_detected_gpu

    @patch('library.sensors.sensors_python.GpuNvidia.is_available', return_value=False)
    @patch('library.sensors.sensors_python.GpuAmd.is_available', return_value=False)
    @patch('library.sensors.sensors_python.GpuIntel.is_available', return_value=True)
    def test_gpu_dispatcher_detects_intel(self, _intel, _amd, _nvidia):
        is_available = sensors_python.Gpu.is_available()

        self.assertTrue(is_available)
        self.assertEqual(sensors_python.DETECTED_GPU, sensors_python.GpuType.INTEL)

    @patch('library.sensors.sensors_python.GpuIntel._linux_intel_cards', return_value=['/sys/class/drm/card0'])
    @patch('library.sensors.sensors_python.GpuIntel._run_intel_gpu_top',
           return_value={'engines': {'Render/3D/0': {'busy': 37.5}}})
    @patch('library.sensors.sensors_python.GpuIntel._load_from_sysfs', return_value=12.5)
    @patch('library.sensors.sensors_python.GpuIntel._temperature_from_hwmon', return_value=61.0)
    def test_gpu_intel_stats_returns_supported_metrics(self, _temp, _sysfs_load, _gpu_top, _cards):
        load, mem_pct, mem_used, mem_total, temperature = sensors_python.GpuIntel.stats()

        self.assertAlmostEqual(load, 37.5)
        self.assertTrue(math.isnan(mem_pct))
        self.assertTrue(math.isnan(mem_used))
        self.assertTrue(math.isnan(mem_total))
        self.assertAlmostEqual(temperature, 61.0)

    @patch('library.sensors.sensors_python.GpuIntel._linux_intel_cards', return_value=['/sys/class/drm/card0'])
    @patch('library.sensors.sensors_python.GpuIntel._run_intel_gpu_top', return_value={})
    @patch('library.sensors.sensors_python.GpuIntel._load_from_sysfs', return_value=15.0)
    @patch('library.sensors.sensors_python.GpuIntel._temperature_from_hwmon', return_value=55.0)
    def test_gpu_intel_stats_falls_back_to_sysfs_load(self, _temp, _sysfs_load, _gpu_top, _cards):
        load, _, _, _, temperature = sensors_python.GpuIntel.stats()

        self.assertAlmostEqual(load, 15.0)
        self.assertAlmostEqual(temperature, 55.0)

    @patch('library.sensors.sensors_python.GpuIntel._read_first_float', side_effect=[math.nan, 150.0, 600.0])
    def test_gpu_intel_sysfs_load_estimate_from_frequency(self, _read):
        load = sensors_python.GpuIntel._load_from_sysfs(['/sys/class/drm/card0'])
        self.assertAlmostEqual(load, 25.0)

    @patch('library.sensors.sensors_python.GpuIntel._linux_intel_cards', return_value=['/sys/class/drm/card0'])
    @patch('library.sensors.sensors_python.GpuIntel._frequency_from_sysfs', return_value=1450.0)
    def test_gpu_intel_frequency_prefers_sysfs(self, _freq, _cards):
        self.assertEqual(sensors_python.GpuIntel.frequency(), 1450.0)

    @patch('library.sensors.sensors_python.GpuIntel._linux_intel_cards', return_value=[])
    def test_gpu_intel_unavailable_returns_nan(self, _cards):
        load, mem_pct, mem_used, mem_total, temperature = sensors_python.GpuIntel.stats()

        self.assertTrue(math.isnan(load))
        self.assertTrue(math.isnan(mem_pct))
        self.assertTrue(math.isnan(mem_used))
        self.assertTrue(math.isnan(mem_total))
        self.assertTrue(math.isnan(temperature))


if __name__ == '__main__':
    unittest.main()
