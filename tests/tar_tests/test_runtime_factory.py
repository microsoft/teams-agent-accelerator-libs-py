import unittest
from unittest.mock import patch, MagicMock
from agent_runtime import (
    RuntimeFactory,
    ZMQ_Runtime,
    ZMQRuntime,
    IRuntime
)

class TestRuntimeFactory(unittest.TestCase):

    def setUp(self):
        # Reset the supported runtimes before each test
        RuntimeFactory._supported_runtimes = {}
        RuntimeFactory._initialize()

    def test_get_zmq_runtime(self):
        runtime = RuntimeFactory.get_runtime(ZMQ_Runtime)
        self.assertIsInstance(runtime, ZMQRuntime)

    def test_get_unknown_runtime(self):
        with self.assertRaises(ValueError):
            RuntimeFactory.get_runtime("UnknownRuntime")

    def test_register_custom_runtime(self):
        class CustomRuntime(IRuntime):
            def register(self, actor):
                pass
            def get_new_msg_receiver(self):
                pass
            def connect(self):
                pass
            def disconnect(self):
                pass
            def find_by_topic(self, topic):
                pass
            def find_by_name(self, name):
                pass
            def find_termination(self):
                pass
            def find_by_name_regex(self, name_regex):
                pass

        custom_runtime = CustomRuntime()
        RuntimeFactory.register_runtime("CustomRuntime", custom_runtime)

        retrieved_runtime = RuntimeFactory.get_runtime("CustomRuntime")
        self.assertIs(retrieved_runtime, custom_runtime)

    @patch('autogencap.runtime_factory.ZMQRuntime')
    def test_initialize(self, mock_zmq_runtime):
        # Reset the supported runtimes
        RuntimeFactory._supported_runtimes = {}

        # Create a mock ZMQRuntime instance
        mock_instance = MagicMock()
        mock_zmq_runtime.return_value = mock_instance

        # Call _initialize
        RuntimeFactory._initialize()

        # Check if ZMQRuntime was registered
        self.assertIn(ZMQ_Runtime, RuntimeFactory._supported_runtimes)
        self.assertIs(RuntimeFactory._supported_runtimes[ZMQ_Runtime], mock_instance)

if __name__ == '__main__':
    unittest.main()
