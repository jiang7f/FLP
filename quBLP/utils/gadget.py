import sys

main_module = sys.modules['__main__']
should_print = getattr(main_module, 'should_print', False)

def iprint(*args, **kwargs):
    if should_print:
        print(*args, **kwargs)
