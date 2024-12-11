import time
from typing import List

import zmq

from .actor import Actor
from .actor_connector import IActorConnector
from .actor_runtime import IMessageReceiver, IRuntime
from .broker import Broker
from .constants import Termination_Topic
from .debug_log import Debug, Warn
from .proto.CAP_pb2 import ActorInfo, ActorInfoCollection
from .zmq_actor_connector import ZMQActorConnector


class ZMQRuntime(IRuntime):
    def __init__(self, start_broker: bool = True):
        # Initialize the ZMQRuntime
        self.local_actors = {}  # Dictionary to store local actors
        self._context: zmq.Context = zmq.Context()  # ZMQ context for creating sockets
        self._start_broker: bool = start_broker  # Flag to determine if broker should be started
        self._broker: Broker = None  # Broker instance
        self._directory_svc = None  # Directory service instance
        self._log_name = self.__class__.__name__  # Logger name

    def __str__(self):
        # String representation of the ZMQRuntime
        # This provides a quick overview of the runtime's state
        return f" \
{self._log_name}\n \
is_broker: {self._broker is not None}\n \
is_directory_svc: {self._directory_svc is not None}\n \
local_actors: {self.local_actors}\n"

    def _init_runtime(self):
        # Initialize the runtime components (broker and directory service)
        if self._start_broker and self._broker is None:
            # Start the broker if it hasn't been started yet
            self._broker = Broker(self._context)
            if not self._broker.start():
                # If broker fails to start, disable future attempts
                self._start_broker = False
                self._broker = None
        if self._directory_svc is None:
            # Initialize the directory service if it hasn't been created yet
            from .zmq_directory_svc import ZMQDirectorySvc
            self._directory_svc = ZMQDirectorySvc(self._context)
            self._directory_svc.start(self)
        time.sleep(0.25)  # Allow time for initialization of components

    def register(self, actor: Actor):
        # Register an actor with the runtime
        self._init_runtime()  # Ensure runtime is initialized before registering
        self._directory_svc.register_actor_by_name(actor.actor_name)  # Register with directory service
        self.local_actors[actor.actor_name] = actor  # Add to local actors dictionary
        actor.on_start(self)  # Notify actor that it has started
        Debug(self._log_name, f"{actor.actor_name} registered in the network.")

    def get_new_msg_receiver(self) -> IMessageReceiver:
        # Create and return a new message receiver
        # This is used by actors to receive messages
        from .zmq_msg_receiver import ZMQMsgReceiver
        return ZMQMsgReceiver(self._context)

    def connect(self):
        # Connect all registered actors to the network
        self._init_runtime()  # Ensure runtime is initialized
        for actor in self.local_actors.values():
            actor.on_connect()  # Notify each actor to connect

    def disconnect(self):
        # Disconnect all actors and stop runtime services
        for actor in self.local_actors.values():
            actor.disconnect_network(self)  # Disconnect each actor
        if self._directory_svc:
            self._directory_svc.stop()  # Stop the directory service
            self._directory_svc = None  # Reset directory service
        if self._broker:
            self._broker.stop()  # Stop the broker
            self._broker = None  # Reset broker
        self.local_actors.clear()  # Clear the local actors dictionary

    def find_by_topic(self, topic: str) -> IActorConnector:
        # Find an actor connector by topic
        # This creates a new connector for the given topic
        return ZMQActorConnector(self._context, topic)

    def find_by_name(self, name: str) -> IActorConnector:
        # Find an actor connector by name
        actor_info: ActorInfo = self._directory_svc.lookup_actor_by_name(name)
        if actor_info is None:
            Warn(self._log_name, f"{name}, not found in the network.")
            return None
        Debug(self._log_name, f"[{name}] found in the network.")
        return self.find_by_topic(name)  # Use the name as the topic

    def find_termination(self) -> IActorConnector:
        # Find the termination actor connector
        # This is a special actor used for termination signals
        termination_topic: str = Termination_Topic
        return self.find_by_topic(termination_topic)

    def find_by_name_regex(self, name_regex) -> List[ActorInfo]:
        # Find actors by name using a regular expression
        actor_info: ActorInfoCollection = self._directory_svc.lookup_actor_info_by_name(name_regex)
        if actor_info is None:
            Warn(self._log_name, f"{name_regex}, not found in the network.")
            return None
        Debug(self._log_name, f"[{name_regex}] found in the network.")
        actor_list = []
        for actor in actor_info.info_coll:
            actor_list.append(actor)
        return actor_list  # Return a list of matching actors
