def nats(x):
    yield x
    yield from nats(x+1)







