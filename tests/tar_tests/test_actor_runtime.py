import unittest
from unittest.mock import MagicMock
import zmq
from agent_runtime import (
    IMsgActor, 
    IMessageReceiver, 
    IRuntime,
    ZMQActorConnector,
    ActorInfo
)

class TestIMsgActor(unittest.TestCase):
    def test_abstract_methods(self):
        methods = ['on_connect', 'on_txt_msg', 'on_bin_msg', 'on_start', 'stop', 'dispatch_message']
        for method in methods:
            self.assertTrue(hasattr(IMsgActor, method))
            self.assertTrue(callable(getattr(IMsgActor, method)))

class TestIMessageReceiver(unittest.TestCase):
    def test_abstract_methods(self):
        methods = ['init', 'get_message', 'stop']
        for method in methods:
            self.assertTrue(hasattr(IMessageReceiver, method))
            self.assertTrue(callable(getattr(IMessageReceiver, method)))

class TestIRuntime(unittest.TestCase):
    def test_abstract_methods(self):
        methods = ['register', 'get_new_msg_receiver', 'connect', 'disconnect', 
                   'find_by_topic', 'find_by_name', 'find_termination', 'find_by_name_regex']
        for method in methods:
            self.assertTrue(hasattr(IRuntime, method))
            self.assertTrue(callable(getattr(IRuntime, method)))

class ConcreteRuntime(IRuntime):
    def register(self, actor):
        pass
    def get_new_msg_receiver(self):
        return MagicMock()
    def connect(self):
        pass
    def disconnect(self):
        pass
    def find_by_topic(self, topic):
        return ZMQActorConnector(zmq.Context(), topic)
    def find_by_name(self, name):
        return ZMQActorConnector(zmq.Context(), name)
    def find_termination(self):
        return ZMQActorConnector(zmq.Context(), "termination")
    def find_by_name_regex(self, name_regex):
        return [ActorInfo(name="test_actor")]

class TestConcreteRuntime(unittest.TestCase):
    def setUp(self):
        self.runtime = ConcreteRuntime()

    def test_register(self):
        actor = MagicMock()
        self.runtime.register(actor)  # Should not raise any exception

    def test_get_new_msg_receiver(self):
        receiver = self.runtime.get_new_msg_receiver()
        self.assertIsInstance(receiver, MagicMock)

    def test_connect_disconnect(self):
        self.runtime.connect()  # Should not raise any exception
        self.runtime.disconnect()  # Should not raise any exception

    def test_find_by_topic(self):
        connector = self.runtime.find_by_topic("test_topic")
        self.assertIsInstance(connector, ZMQActorConnector)

    def test_find_by_name(self):
        connector = self.runtime.find_by_name("test_name")
        self.assertIsInstance(connector, ZMQActorConnector)

    def test_find_termination(self):
        connector = self.runtime.find_termination()
        self.assertIsInstance(connector, ZMQActorConnector)

    def test_find_by_name_regex(self):
        actors = self.runtime.find_by_name_regex("test.*")
        self.assertIsInstance(actors, list)
        self.assertEqual(len(actors), 1)
        self.assertIsInstance(actors[0], ActorInfo)
        self.assertEqual(actors[0].name, "test_actor")

if __name__ == '__main__':
    unittest.main()
