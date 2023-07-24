# coding=utf-8
"""
.. moduleauthor:: Shuhua Gao

The module :mod:`entity` defines the core data structures used in GEP, including the gene, the chromosome, the
K-expression and the expression tree. We use primitives (functions and terminals) to build genes and then compose
a chromosome with one or multiple genes. Refer to the :mod:`geppy.core.symbol` module for information on primitives.

A chromosome composed of only one gene is called a *monogenic* chromosome, while one composed of more than one genes
is named a *multigenic* chromosome. A multigenic chromosome can be assigned a linking function to combine the results
from multiple genes into a single result.
"""
import copy
from ..tools.generator import *
from ..core.symbol import Function, RNCTerminal, Terminal


_DEBUG = False


class KExpression(list):
    """
    Class representing the K-expression, or the open reading frame (ORF), of a gene in GEP. A K-expression is actually
    a linear form of an expression tree obtained by level-order traversal.
    """
    def __init__(self, content):
        """
        Initialize a K-expression.

        :param content: iterable, each element is a primitive (function or terminal)
        """
        list.__init__(self, content)

    def __str__(self):
        """
        Get a string representation of this expression by joining the name of each primitive.
        :return:
        """
        return '[' + ', '.join(ele.name for ele in self) + ']'

    def __repr__(self):
        return str(self.__class__) + '[{}]'.format(', '.join(repr(p) for p in self))


class ExpressionTree:
    """
    Class representing an expression tree (ET) in GEP, which may be obtained by translating a K-expression, a
    gene, or a chromosome, i.e., genotype-phenotype mapping.
    """
    def __init__(self, root):
        """
        Initialize a tree with the given *root* node.

        :param root: :class:`ExpressionTree.Node`, the root node
        """
        self._root = root

    @property
    def root(self):
        """
        Get the root node of this expression tree.
        """
        return self._root

    class Node:
        """
        Class representing a node in the expression tree. Each node has a variable number of children, depending on
        the arity of the primitive at this node.
        """
        def __init__(self, name):
            self._children = []
            self._name = name

        @property
        def children(self):
            """
            Get the children of this node.
            """
            return self._children

        @property
        def name(self):
            """
            Get the name (label) of this node.
            """
            return self._name

    @classmethod
    def from_genotype(cls, genome):
        """
        Create an expression tree by translating *genome*, which may be a K-expression, a gene, or a chromosome.

        :param genome: :class:`KExpression`, :class:`Gene`, or :class:`Chromosome`, the genotype of an individual
        :return: :class:`ExpressionTree`, an expression tree
        """
        if isinstance(genome, Gene):
            return cls._from_kexpression(genome.kexpression)
        elif isinstance(genome, KExpression):
            return cls._from_kexpression(genome)
        elif isinstance(genome, Chromosome):
            if len(genome) == 1:
                return cls._from_kexpression(genome[0].kexpression)
            sub_trees = [cls._from_kexpression(gene.kexpression) for gene in genome]
            # combine the sub_trees with the linking function
            root = cls.Node(genome.linker.__name__)
            root.children[:] = sub_trees
            return cls(root)
        raise TypeError('Only an argument of type KExpression, Gene, and Chromosome is acceptable. The provided '
                        'genome type is {}.'.format(type(genome)))

    @classmethod
    def _from_kexpression(cls, expr):
        """
        Create an expression tree from a K-expression.

        :param expr: a K-expression
        :return: :class:`ExpressionTree`, an expression tree
        """
        if len(expr) == 0:
            return None
        # first build a node for each primitive
        nodes = [cls.Node(p.name) for p in expr]
        # connect each node to its children if any
        i = 0
        j = 0
        while i < len(nodes):
            for _ in range(expr[i].arity):
                j += 1
                nodes[i].children.append(nodes[j])
            i += 1
        return cls(nodes[0])


class Gene(list):
    """
    A single gene in GEP, which is a fixed-length linear sequence composed of functions and terminals.
    """
    # TODO (Ryan Heminway) added guided here
    def __init__(self, pset, head_length, guided=False):
        """
        Instantiate a gene.
        :param head_length: length of the head domain
        :param pset: a primitive set including functions and terminals for genome construction.

        Supposing the maximum arity of functions in *pset* is *max_arity*, then the tail length is automatically
        determined to be ``tail_length = head_length * (max_arity - 1) + 1``. The genome, i.e., list of symbols in
        the instantiated gene is formed randomly from *pset*.
        """
        self._head_length = head_length
        genome = generate_genome(pset, head_length, guided)
        list.__init__(self, genome)

    @classmethod
    def from_genome(cls, genome, head_length):
        """
        Build a gene directly from the given *genome*.

        :param genome: iterable, a list of symbols representing functions and terminals
        :param head_length: length of the head domain
        :return: :class:`Gene`, a gene
        """
        g = super().__new__(cls)
        super().__init__(g, genome)
        g._head_length = head_length
        return g

    def __str__(self):
        """
        Return the expression in a human readable string, which is also a legal python code that can be evaluated.

        :return: string form of the expression
        """
        expr = self.kexpression
        i = len(expr) - 1
        while i >= 0:
            if expr[i].arity > 0:  # a function
                f = expr[i]
                args = []
                for _ in range(f.arity):
                    ele = expr.pop()
                    if isinstance(ele, str):
                        args.append(ele)
                    else:  # a terminal or an ephemeral class
                        args.append(ele.format())
                expr[i] = f.format(*reversed(args))  # replace the operator with its result (str)
            i -= 1

        # the final result is at the root. It may happen that the K-expression contains only one terminal at its root.
        return expr[0] if isinstance(expr[0], str) else expr[0].format()

    @property
    def kexpression(self):
        """Get the K-expression of type :class:`KExpression` represented by this gene.
        """
        # get the level-order K expression
        expr = KExpression([self[0]])
        i = 0
        j = 1
        while i < len(expr):
            for _ in range(expr[i].arity):
                expr.append(self[j])
                j += 1
            i += 1
        return expr

    @property
    def head_length(self):
        """
        Get the length of the head domain.
        """
        return self._head_length

    @property
    def tail_length(self):
        """
        Get the length of the tail domain.
        """
        return len(self) - self.head_length

    @property
    def max_arity(self):
        """
        Get the max arity of the functions in the primitive set with which the gene is built.
        """
        return (len(self) - 1) // self.head_length

    @property
    def head(self):
        """
        Get the head domain of the gene.
        """
        return self[0: self.head_length]

    @property
    def tail(self):
        """
        Get the tail domain of the gene.
        """
        return self[self.head_length: self.head_length + self.tail_length]

    def __repr__(self):
        """
        Return a string representation of a gene including all the symbols. Mainly for debugging purpose.

        :return: a string
        """
        info = [str(p) for p in self]
        return '{} ['.format(self.__class__) + ', '.join(info) + ']'


class GeneDc(Gene):
    """
    Class represents a gene with an additional Dc domain to handle numerical constants in GEP.

    The basic :class:`Gene`
    has two domains, a head and a tail, while this :class:`GeneDc` class introduces another domain called *Dc*
    after the tail. The length of the Dc domain is equal to the length of the tail domain. The Dc domain only stores
    the indices of numbers present in a separate array :meth:`rnc_array`, which collects a group of candidate
    random numerical constants (RNCs). Thus, each :class:`GeneDc` instance comes with a *rnc_array*, which are
    generally different among different instances.

    In addition to the operators of the basic gene expression algorithm (mutation, inversion, transposition, and
    recombination), Dc-specific operators in GEP-RNC are also created to better evolve the Dc domain and to manipulate
    the attached set of constants *rnc_array*. Such operators are all suffixed with '_dc' in
    :mod:`~geppy.tools.crossover` and :mod:`~geppy.tools.mutation` modules, for example, the method
    :func:`~geppy.tools.mutation.invert_dc`.

    The *rnc_array* associated with each :class:`GeneDc` instance can be provided explicitly when creating
    a :class:`GeneDc` instance. See :meth:`from_genome`. Or more generally, a
    random number generator *rnc_gen* can be provided, with which a specified number of RNCs are generated during the
    creation of gene instances.

    A special terminal of type :class:`~geppy.core.symbol.TerminalRNC` is used internally to represent the RNCs.

    .. note::
        To create an effective :class:`GeneDc` gene, the primitive set should contain at least one
        :class:`~geppy.core.symbol.TerminalRNC` terminal. See :meth:`~geppy.core.symbol.PrimitiveSet.add_rnc`
        for details.

    Refer to Chapter 5 of [FC2006]_ for more knowledge about GEP-RNC.
    """
    def __init__(self, pset, head_length, rnc_gen, rnc_array_length):
        """
        Initialize a gene with a Dc domain.

        :param head_length: length of the head domain
        :param pset: a primitive set including functions and terminals for genome construction.
        :param rnc_gen: callable, which should generate a random number when called by ``rnc_gen()``.
        :param rnc_array_length: int, number of random numerical constant candidates associated with this gene,
            usually 10 is enough

        Supposing the maximum arity of functions in *pset* is *max_arity*, then the tail length is automatically
        determined to be ``tail_length = head_length * (max_arity - 1) + 1``. The genome, i.e., list of symbols in
        the instantiated gene is formed randomly from *pset*. The length of Dc domain is equal to *tail_length*, i.e.,
        the tail domain and the Dc domain share the same length.
        """
        # first generate the gene without Dc
        super().__init__(pset, head_length)
        t = head_length * (pset.max_arity - 1) + 1
        d = t
        # complement it with a Dc domain
        self._rnc_gen = rnc_gen
        dc = generate_dc(rnc_array_length, dc_length=d)
        self.extend(dc)
        # generate the rnc array
        self._rnc_array = [self._rnc_gen() for _ in range(rnc_array_length)]

    @classmethod
    def from_genome(cls, genome, head_length, rnc_array):
        """
        Build a gene directly from the given *genome* and the random numerical constant (RNC) array *rnc_array*.

        :param genome: iterable, a list of symbols representing functions and terminals (especially the special RNC
            terminal of type :class:`~geppy.core.symbol.TerminalRNC`). This genome should have three
            domains: head, tail and Dc. The Dc domain should be composed only of integers representing indices into the
            *rnc_array*.
        :param head_length: length of the head domain. The length of the tail and Dc domain should follow the rule and
            can be determined from the head_length: ``dc_length = tail_length = (len(genome) - head_length) / 2``.
        :param rnc_array: the RNC array associated with the gene, which contains random constant candidates
        :return: :class:`GeneDc`, a gene
        """
        g = super().from_genome(genome, head_length)  # the genome is copied and the head-length is fixed
        g._rnc_array = copy.deepcopy(rnc_array)
        return g

    @property
    def dc_length(self):
        """
        Get the length of the Dc domain, which is equal to :meth:`GeneDc.tail_length`
        """
        return self.tail_length

    @property
    def rnc_array(self):
        """
        Get the random numerical array (RNC) associated with this gene.
        """
        return self._rnc_array

    @property
    def tail_length(self):
        """
        Get the tail domain length.
        """
        return (len(self) - self.head_length) // 2

    @property
    def max_arity(self):
        """
        Get the max arity of functions in the primitive set used to build this gene.
        """
        # determine from the head-tail-Dc length relations
        return (len(self) - 2 + self.head_length) // (2 * self.head_length)

    @property
    def dc(self):
        """
        Get the Dc domain of this gene.
        :return:
        """
        return self[self.head_length + self.tail_length: self.head_length + self.tail_length + self.dc_length]

    # def __deepcopy__(self, memodict):
    #     return self.__class__.from_genome(self, self.head_length, self.rnc_array)

    def __str__(self):
        """
        Return the expression in a human readable string, which is also a legal python code that can be evaluated.
        The special terminal representing a RNC will be replaced by their true values retrieved from the array
        :meth:`rnc_array`.

        :return: string form of the expression
        """
        expr = self.kexpression
        n_total_rncs = sum(1 if isinstance(p, RNCTerminal) else 0 for p in expr)  # how many RNCs in total?
        n_encountered_rncs = 0  # how many RNCs we have encountered in reverse order?
        i = len(expr) - 1
        while i >= 0:
            if expr[i].arity > 0:  # a function
                f = expr[i]
                args = []
                for _ in range(f.arity):
                    ele = expr.pop()
                    if isinstance(ele, str):
                        args.append(ele)
                    else:  # a terminal or a RNC terminal
                        if isinstance(ele, RNCTerminal):
                            # retrieve its true value
                            which = n_total_rncs - n_encountered_rncs - 1
                            index = self.dc[which]
                            value = self.rnc_array[index]
                            args.append(str(value))
                            n_encountered_rncs += 1
                        else:   # a normal terminal
                            args.append(ele.format())
                expr[i] = f.format(*reversed(args))  # replace the operator with its result (str)
            i -= 1

        # the final result is at the root
        if isinstance(expr[0], str):
            return expr[0]
        if isinstance(expr[0], RNCTerminal):  # only contains a single RNC, let's retrieve its value
            return str(self.rnc_array[self.dc[0]])
        return expr[0].format()    # only contains a normal terminal

    @property
    def kexpression(self):
        """
        Get the K-expression of type :class:`KExpression` represented by this gene. The involved RNC terminal will be
        replaced by a constant terminal with its value retrived from the :meth:`rnc_array` according to the GEP-RNC
        algorithm.
        """
        n_rnc = 0

        def convert_RNC(p):
            nonlocal n_rnc
            if isinstance(p, RNCTerminal):
                index = self.dc[n_rnc]
                value = self.rnc_array[index]
                n_rnc += 1
                t = Terminal(str(value), value)
                return t
            return p

        # level-order
        expr = KExpression([convert_RNC(self[0])])
        i = 0
        j = 1
        while i < len(expr):
            for _ in range(expr[i].arity):
                expr.append(convert_RNC(self[j]))
                j += 1
            i += 1
        return expr

    def __repr__(self):
        return super().__repr__() + ', rnc_array=[' + ', '.join(str(num) for num in self.rnc_array) + ']'



# ----------------- RYAN HEMINWAY ADDITION (start) ------------------- #

class GeneNN(Gene):
    """
    Class represents a gene with additional Dw and Dt domains to handle weights and thresholds in GEP-NN.

    The basic :class:`Gene`
    has two domains, a head and a tail, while this :class:`GeneNN` class introduces another two domains called *Dw*
    and *Dt*, respectively, after the tail. The length of the Dw domain is equal to the length of the head, h, 
    multiplied by the maximum arity in the function set, n. The length of the Dt domain is the equal to the length 
    of the head, h. The Dw and Dt domains only store the indices of numbers present in separate arrays :meth:`dw_array`
    and :meth:`dt_array`, which collect a group of candidate random numerical constants (RNCs). Thus, each :class:`GeneNN` 
    instance comes with a *dw_array* and a *dt_array*, which are generally different among different instances.

    In addition to the operators of the basic gene expression algorithm (mutation, inversion, transposition, and
    recombination), NN-specific operators in GEP-NN are also created to better evolve the Dw and Dt domains and to
    manipulate the attached set of constants *dw_array* and *dt_array*. 
    
    Such operators are all suffixed with '_dw' or '_dt' in
    :mod:`~geppy.tools.crossover` and :mod:`~geppy.tools.mutation` modules, for example, the method
    :func:`~geppy.tools.mutation.transpose_dw`.  

    The *dw_array* and *dt_array* associated with each :class:`GeneNN` instance can be provided explicitly when creating
    a :class:`GeneNN` instance. See :meth:`from_genome`. Or more generally, a random number generator *rnc_gen* can be 
    provided, with which a specified number of RNCs are generated during the creation of gene instances.

    """
    def __init__(self, pset, head_length, dw_rnc_gen, dw_rnc_array_length, dt_rnc_gen, dt_rnc_array_length, func_head):
        """
        Initialize a gene with a Dc domain.

        :param head_length: length of the head domain
        :param pset: a primitive set including functions and terminals for genome construction.
        :param dw_rnc_gen: callable, which should generate a random number when called by ``dw_rnc_gen()``.
        :param dw_rnc_array_length: int, number of random numerical constant candidates 
                associated with this gene's Dw domain, usually 10 is enough
        :param dt_rnc_gen: callable, which should generate a random number when called by ``dt_rnc_gen()``.
        :param dt_rnc_array_length: int, number of random numerical constant candidates 
                associated with this gene's Dt domain, usually 10 is enough
        :pararm func_head: boolean, indicating whether head should only contain Functions

        Supposing the maximum arity of functions in *pset* is *max_arity*, then the tail length is automatically
        determined to be ``tail_length = head_length * (max_arity - 1) + 1``. The genome, i.e., list of symbols in
        the instantiated gene is formed randomly from *pset*. The length of Dw domain is equal to (*head_length* * *max_arity*),
        and the length of the Dt domain is the same as the *head_length*. 
        """
        # first generate the gene without Dw or Dt
        super().__init__(pset, head_length, func_head)
        self._max_arity = pset.max_arity
        # tail length
        t = head_length * (pset.max_arity - 1) + 1
        # length of Dw is h * n, where n is max arity 
        wLen = head_length * (pset.max_arity)
        # length of Dt is h
        tLen = head_length
        # complement it with a Dw domain and Dt domain
        self._dw_rnc_gen = dw_rnc_gen
        dw = generate_dc(dw_rnc_array_length, dc_length=wLen)
        self.extend(dw)
        self._dt_rnc_gen = dt_rnc_gen
        dt = generate_dc(dt_rnc_array_length, dc_length=tLen)
        self.extend(dt)
        # generate the rnc arrays
        self._dw_rnc_array = [self._dw_rnc_gen() for _ in range(dw_rnc_array_length)]
        self._dt_rnc_array = [self._dt_rnc_gen() for _ in range(dt_rnc_array_length)]


    @classmethod
    def from_genome(cls, genome, head_length, max_arity, dw_rnc_array, dt_rnc_array):
        """
        Build a gene directly from the given *genome* and the random numerical constant (RNC) array *rnc_array*.

        :param genome: iterable, a list of symbols representing functions and terminals. This genome should have four
            domains: head, tail, Dw, and Dt. The Dw and Dt domains should be composed only of integers representing indices into the
            *dw_rnc_array* and *dt_rnc_array* respectively.
        :param head_length: length of the head domain. The length of the Dt domain is equal to the head domain. The
            length of the Dw domain is: ``Dw_length = head_length * max_arity``. The length of the tail can be 
            determined as: ``tail_length = (len(genome) - (2 * head_length) - (Dw_length))``.
        :param max_arity: maximum arity in the primitive set for this gene 
        :param dw_rnc_array: the RNC array associated with the Dw domain of the gene, which contains random constant candidates
        :param dt_rnc_array: the RNC array associated with the Dt domain of the gene, which contains random constant candidates
        :return: :class:`GeneNN`, a gene
        """
        g = super().from_genome(genome, head_length)  # the genome is copied and the head-length is fixed
        g._dw_rnc_array = copy.deepcopy(dw_rnc_array)
        g._dt_rnc_array = copy.deepcopy(dt_rnc_array)
        return g

    @property
    def dw_length(self):
        """
        Get the length of the Dw domain
        """
        return self.head_length * self._max_arity
    
    @property
    def dt_length(self):
        """
        Get the length of the Dt domain
        """
        return self.head_length

    @property
    def dw_rnc_array(self):
        """
        Get the random numerical array (RNC) associated with this gene's Dw domain.
        """
        return self._dw_rnc_array
    
    @property
    def dt_rnc_array(self):
        """
        Get the random numerical array (RNC) associated with this gene's Dt domain.
        """
        return self._dt_rnc_array

    @property
    def tail_length(self):
        """
        Get the tail domain length.
        """
        return (len(self) - self.head_length - self.dt_length - self.dw_length)
    
    @property
    def max_arity(self):
        """
        Get the max arity of functions in the primitive set used to build this gene.
        """
        return self._max_arity

    @property
    def dw(self):
        """
        Get the Dw domain of this gene.
        :return:
        """
        return self[self.head_length + self.tail_length: self.head_length + self.tail_length + self.dw_length]
    
    @property
    def dt(self):
        """
        Get the Dw domain of this gene.
        :return:
        """
        return self[self.head_length + self.tail_length + self.dw_length: len(self)]

    def __str__(self): 
        """
        Return the expression in a human readable string. Functions which require
        weights and thresholds as inputs will have them added, retrieved from
        Dw and Dt respectively. 
    
        :return: string form of the expression
        """
        expr = self.kexpression
        w_i = self.dw_length - 1
        t_i = self.dt_length - 1
        i = len(expr) - 1
        while i >= 0:
            if expr[i].arity > 0:  # a function
                f = expr[i]
                args = []
                # Pop node inputs
                for _ in range(f.arity):
                    ele = expr.pop()
                    if isinstance(ele, str):
                        args.append(ele)
                    else:  # a terminal
                        args.append(ele.format())
                        
                # reverse node inputs before including weights/threshold
                args = [*(reversed(args))]
                        
                # Pop node weights
                weights = []
                for _ in range(f.arity):
                    index = self.dw[w_i]
                    value = self._dw_rnc_array[index]
                    weights.append(value)
                    w_i = w_i - 1
                args.append(str(weights))
                
                # Pop node threshold
                index = self.dt[t_i]
                value = self._dt_rnc_array[index]
                t_i = t_i - 1
                args.append(str(value))
                expr[i] = f.format(*args)  # replace the operator with its result (str)
            i -= 1

        # the final result is at the root
        if isinstance(expr[0], str):
            return expr[0]
        if isinstance(expr[0], RNCTerminal):  # only contains a single RNC, let's retrieve its value
            return str(self.rnc_array[self.dc[0]])
        return expr[0].format()    # only contains a normal terminal

    # @property
    # def kexpression(self):
    #     """
    #     Get the K-expression of type :class:`KExpression` represented by this gene.
        
    #     (NOTE Ryan Heminway 5/23/23) Expression will not indicate thresholds
    #     or weights. 
    #     """
    #     return Gene.kexpression

    def __repr__(self):
        return super().__repr__() + ', dw_rnc_array=[' + ', '.join(str(num) for num in self.dw_rnc_array) + ']' + ', dt_rnc_array=[' + ', '.join(str(num) for num in self.dt_rnc_array) + ']' 


# ------------------ RYAN HEMINWAY ADDITION (end) -------------------- #



def _default_linker(*args): 
    return tuple(args)


class Chromosome(list):
    """
    Individual representation in gene expression programming (GEP), which may contain one or more genes. Note that in a
    multigenic chromosome, all genes must share the same head length and the same primitive set. Each element of
    :class:`Chromosome` is of type :class:`Gene`.
    """

    def __init__(self, gene_gen, n_genes, linker=None):
        """
        Initialize an individual in GEP.

        :param gene_gen: callable, a gene generator, i.e., ``gene_gen()`` should return a gene
        :param n_genes: number of genes
        :param linker: callable, a linking function

        .. note::
            If a linker is specified, then it must accept *n_genes* arguments, each produced by one gene. If the
            *linker* parameter is the default value ``None``, then for a monogenic chromosome, no linking is applied,
            while for a multigenic chromosome, the ``tuple`` function is used, i.e., a tuple of values of all genes 
            is returned.
        """
        list.__init__(self, (gene_gen() for _ in range(n_genes)))
        self._linker = linker

    @classmethod
    def from_genes(cls, genes, linker=None):
        """
        Create a chromosome instance by providing a list of genes directly. The number of genes is implicitly specified
        by ``len(genes)``.

        :param genes: iterable, a list of genes
        :param linker: callable, a linking function. Refer to :meth:`Chromosome.__init__` for more details.
        :return: :class:`Chromosome`, a chromosome

        .. caution:
            Every gene of type :class:`Gene` in *genes* should be an independent object. That is, for any two genes
            *g1* and *g2*, the assertion ``g1 is g1`` should return ``False``. Otherwise, unexpected results may happen
            in following evolution.

        """
        c = super().__new__(cls)
        super().__init__(c, genes)
        c._linker = linker
        return c

    @property
    def head_length(self):
        """
        Get the length of the head domain. All genes in a chromosome should have the same head length.
        """
        return self[0].head_length

    @property
    def tail_length(self):
        """
        Get the length of the tail domain. All genes in a chromosome should have the same tail length.
        """
        return self[0].tail_length

    @property
    def max_arity(self):
        """
        Get the max arity of the functions in the primitive set with which all genes are built. Note that all genes
        in a chromosome should share the same primitive set.
        """
        return self[0].max_arity

    def __str__(self):
        """
        Return the expressions in a human readable string.
        """
        if self.linker is not None:
            return '{0}(\n\t{1}\n)'.format(self.linker.__name__, ',\n\t'.join(str(g) for g in self))
        else:
            assert len(self) == 1
            return str(self[0])

    @property
    def kexpressions(self):
        """
        Get a list of K-expressions for all the genes in this chromosome.
        """
        return [g.kexpression for g in self]

    @property
    def linker(self):
        """
        Get the linking function. 

        .. note:
            If linker is `None`, then the actual linking function during evaluation of an individual
            returns the value of the single gene for a single-gene (monogenic) chromosome or a tuple of values of all 
            genes for a multi-gene chromosome.
        """
        # the default linker for a multi-gene chromosome
        if self._linker is None and len(self) > 1:
            return _default_linker
        return self._linker

    def __repr__(self):
        """
        Return a string representation of the chromosome including all the symbols in all the genes.
        Mainly for debugging purpose.

        :return: a string
        """
        return '{}[\n\t'.format(self.__class__) + ',\n\t'.join(repr(g) for g in self) + \
               '\n], linker={}'.format(self.linker)


