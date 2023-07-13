import unittest

from src import hello_world


class TestSayHello(unittest.TestCase):
    def test_say_hello(self):
        self.assertEqual(hello_world.say_hello("World"), "Hello, World!")
