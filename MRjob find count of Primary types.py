#since the primary type is 6th field in csv we use [5] to find primary type
from mrjob.job import MRJob
  
class MRCountSum(MRJob):

    def mapper(self, _, line):
        line = line.strip() # remove leading and trailing whitespace
        yield line.split(',')[5],1#primary type

    def combiner(self, key, values):
        yield key, sum(values)

    def reducer(self, key, values):
        yield key, sum(values)

if name == 'main':
    MRCountSum.run()