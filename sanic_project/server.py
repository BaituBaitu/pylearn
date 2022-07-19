#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pylearn 

@Author  ：BaituBaitu
@Date    ：2022/7/18 16:24 
'''

"""
experimenmts with SANIC
working with regular interpreter

"""
from sanic import Sanic
from sanic.response import text

app = Sanic("MyHelloWorldApp")

@app.get("/")
async def hello_world(request):
    return text("Hello, world.")
