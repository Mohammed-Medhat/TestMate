class BoundedStack:

    def __init__(self, max_size: int=100):
        self.items = []
        self.max_size = max_size

    def push(self, item):
        if len(self.items) >= self.max_size:
            raise Exception('Stack overflow')
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            raise Exception('Stack underflow')