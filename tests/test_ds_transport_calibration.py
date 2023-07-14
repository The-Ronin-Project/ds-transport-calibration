import unittest

from src import ds_transport_calibration


class TestSayHello(unittest.TestCase):
    def test_say_hello(self):
        self.assertEqual(ds_transport_calibration.say_hello("World"), "Hello, World!")
