import unittest
import zmq
from agent_runtime import (
    ZMQRuntime,
    Actor,
    ZMQActorConnector,
    ActorInfo,
    Broker,
    ZMQDirectorySvc,
    RuntimeFactory,
    ZMQ_Runtime,
    IActorConnector
)

class TestZMQRuntime(unittest.TestCase):

    def setUp(self):
        self.runtime = RuntimeFactory.get_runtime(ZMQ_Runtime)

    def tearDown(self):
        self.runtime.disconnect()

    def test_init(self):
        self.assertIsInstance(self.runtime, ZMQRuntime)
        self.assertEqual(self.runtime.local_actors, {})
        self.assertTrue(self.runtime._start_broker)
        self.assertIsNone(self.runtime._broker)
        self.assertIsNone(self.runtime._directory_svc)
        self.assertIsInstance(self.runtime._context, zmq.Context)

    def test_register(self):
        class TestActor(Actor):
            def __init__(self):
                super().__init__("test_actor", "Test Actor")

        test_actor = TestActor()
        self.runtime.register(test_actor)
        self.assertIn("test_actor", self.runtime.local_actors)
        self.assertIsInstance(self.runtime._directory_svc, ZMQDirectorySvc)

    def test_get_new_msg_receiver(self):
        from autogencap.zmq_msg_receiver import ZMQMsgReceiver
        receiver = self.runtime.get_new_msg_receiver()
        self.assertIsInstance(receiver, ZMQMsgReceiver)

    def test_connect(self):
        class TestActor(Actor):
            def __init__(self):
                super().__init__("test_actor", "Test Actor")
            
            def on_connect(self, runtime):
                self.connected = True

        test_actor = TestActor()
        self.runtime.register(test_actor)
        self.runtime.connect()
        self.assertTrue(test_actor.connected)

    def test_disconnect(self):
        self.runtime._init_runtime()
        self.runtime.disconnect()
        self.assertIsNone(self.runtime._broker)
        self.assertIsNone(self.runtime._directory_svc)

    def test_find_by_topic(self):
        connector = self.runtime.find_by_topic("test_topic")
        self.assertIsInstance(connector, ZMQActorConnector)

    def test_find_by_name(self):
        self.runtime._init_runtime()
        self.runtime._directory_svc.register_actor_by_name("test_actor")
        connector = self.runtime.find_by_name("test_actor")
        self.assertIsInstance(connector, IActorConnector)

    def test_find_termination(self):
        connector = self.runtime.find_termination()
        self.assertIsInstance(connector, IActorConnector)

    def test_find_by_name_regex(self):
        self.runtime._init_runtime()
        self.runtime._directory_svc.register_actor_by_name("test_actor1")
        self.runtime._directory_svc.register_actor_by_name("test_actor2")
        result = self.runtime.find_by_name_regex("test.*")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], ActorInfo)
        self.assertIsInstance(result[1], ActorInfo)

if __name__ == '__main__':
    unittest.main()
