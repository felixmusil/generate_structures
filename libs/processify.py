import os
import sys
import traceback
from functools import wraps
from multiprocessing import Process, Queue
from Queue import Empty

description= """
Decorator function processify change a function into an executable. 
Use case: the failure of such processified function will not crash the main python program even in case of a seg fault in C or fortran program.
"""


def processify(timeout=None):
    '''
    :param timeout: in seconds
    :return: 
    '''
    def processify_decorator(func):
        '''Decorator to run a function as a process.
        Be sure that every argument and the return value
        is *pickable*.
        The created process is joined, so the code does not
        run in parallel.
        mostly from https://gist.github.com/schlamar/2311116
        '''

        def process_func(q, *args, **kwargs):
            try:
                ret = func(*args, **kwargs)
            except Exception:
                ex_type, ex_value, tb = sys.exc_info()
                error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
                ret = None
            else:
                error = None

            q.put((ret, error))

        # register original function with different name
        # in sys.modules so it is pickable
        process_func.__name__ = func.__name__ + 'processify_func'
        setattr(sys.modules[__name__], process_func.__name__, process_func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            q = Queue()
            p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
            p.start()
            try:
                ret, error = q.get(block=True,timeout=timeout)
            # in the case of timeout no error is raised
            # and output is None
            except Empty:
                ret = None
                error = None
            p.join()

            if error:
                ex_type, ex_value, tb_str = error
                message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
                raise ex_type(message)

            return ret
        return wrapper
    return processify_decorator

@processify
def test_function():
    return os.getpid()


@processify
def test_deadlock():
    return range(30000)


@processify
def test_exception():
    raise RuntimeError('xyz')


def test():
    print os.getpid()
    print test_function()
    print len(test_deadlock())
    test_exception()

if __name__ == '__main__':
    test()