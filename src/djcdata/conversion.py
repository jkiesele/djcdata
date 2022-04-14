try:
    import datastructures
    from datastructures import *
except ImportError:
    print('datastructure modules not found. Please define TrainData definitions in a datastructures module in your python path')
    exit()
    
class_options=[]
import inspect, sys
for name, obj in inspect.getmembers(sys.modules['datastructures']):
    if inspect.isclass(obj) and 'TrainData' in name:
        class_options.append(obj)

if len(class_options) < 1:
    print('No TrainData classes found. Please define TrainData definitions in a datastructures module and name them TrainData_XYZ.')      

class_options = dict((str(i).split("'")[1].split('.')[-1], i) for i in class_options)
