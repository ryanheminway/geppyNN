# coding=utf-8
"""
.. moduleauthor:: Shuhua Gao

This module :mod:`parser` provides functionality for compiling an individual (a chromosome) in GEP into an executable
lambda function in Python for subsequent fitness evaluation.

.. todo::
    Parse an individual into codes of other languages, such as C++/Java, for deployment in an industrial environment.
"""

import sys


def _compile_gene(g, pset):
    """
    Compile one gene *g* with the primitive set *pset*.
    :return: a function or an evaluated result
    """
    code = str(g)
    if len(pset.input_names) > 0:   # form a Lambda function
        args = ', '.join(pset.input_names)
        code =  'lambda {}: {}'.format(args, code)
    # evaluate the code
    try:
        return eval(code, pset.globals, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("The expression tree generated by GEP is too deep. Python cannot evaluate a tree higher "
                          "than 90. You should try to adopt a smaller head length for the genes, for example, by using"
                          "more genes in a chromosome.").with_traceback(traceback)


def compile_(individual, pset):
    """
    Compile the individual into a Python lambda expression.

    :param individual: :class:`Chromosome`, a chromosome
    :param pset: :class:`PrimitiveSet`, a primitive set
    :return: a function if the primitive set *pset* has any inputs (arguments), which can later be called with
        specific parameter values; otherwise, a numerical result obtained from evaluation.
    """
    fs = [_compile_gene(gene, pset) for gene in individual]
    linker = individual.linker
    if linker is None:  # return the gene itself for a monogenic one or tuple of all genes
        if len(fs) == 1:
            return fs[0]
        else:
            return lambda *x: tuple((f(*x) for f in fs))
    return lambda *x: linker(*(f(*x) for f in fs))

__all__ = ['compile_']
