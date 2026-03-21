"""

"""
from dataclasses import dataclass


@dataclass
class MessageHandler():
    message: str

    def get_message(self):
        """

        :return:
        """
        return self.message

    def set_message(self, message):
        """

        :param message:
        :return:
        """
        self.message = message

    def append_message(self, message: list[str]):
        """

        :param message:
        :return:
        """
        self.message = " ".join(message)
