#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pylearn 
@File    ：helloworld.py
@Author  ：BaituBaitu
@Date    ：2022/7/15 15:06 
"""


# to add a .py file to pylearn repository and push it to Github

class Calculator:
    """ a class declaration

    Methods:
        add(a, b): adddsd
        multiply(a, b): multipliesd

    Examples:
        >>> calc = Calculator()
        >>> calc.add(1, 2)
        [3, None]
        >>> calc.multiply(2, 3)
        6

    """

    def __init__(self):
        self.eggs = 'eggs'

    @staticmethod
    def add(a: float, b: float = 0.0) -> list[float]:
        """

        Args:
            a (float):
            b (float):

        Returns:
            float:
        """
        return [a + b, None]

    def multiply(self, a: float, b) -> float:
        """

        Args:
            a ():
            b ():

        Returns:

        """
        return a * b

    def divide(self, a: float, b: float):
        """

        Args:
            a ():
            b ():

        Returns:
            float:

        # """
        return a / b

    def sub(self, a: float, b: float) -> float:
        """

        Args:
            a ():
            b ():

        Returns:
            float:

        """


        return a - b
