"""Coverage ledger shared by the complete CPU rehearsal and later native loop."""
import math
from .common import require


class Coverage:
    def __init__(self,examples,batch_size):
        require(type(examples) is int and examples>0 and type(batch_size) is int and batch_size>0,'Invalid epoch extent')
        self.expected=examples;self.size=batch_size;self.examples=self.updates=0
    def observe(self,outputs):
        require(type(outputs) is int and outputs==min(self.size,self.expected-self.examples) and outputs>0,
                'Missing, repeated or truncated training batch')
        self.examples+=outputs;self.updates+=1
    def finish(self):
        require(self.examples==self.expected and self.updates==math.ceil(self.expected/self.size),'Incomplete epoch')
        return dict(examples=self.examples,updates=self.updates,last_batch=self.expected%self.size or self.size,epochs=1)
