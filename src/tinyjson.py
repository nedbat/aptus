# A teeny tiny JSON module.
# Ned Batchelder, http://nedbatchelder.com

# Safe eval, from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/364469

import compiler

class UnsafeSourceError(Exception):
    def __init__(self, error, descr=None, node=None):
        self.error = error
        self.descr = descr
        self.node = node
        self.lineno = getattr(node,"lineno",None)
        
    def __repr__(self):
        return "Line %d.  %s: %s" % (self.lineno, self.error, self.descr)
    __str__ = __repr__    
           
class SafeEval(object):
    
    def visit(self, node, **kw):
        cls = node.__class__
        meth = getattr(self, 'visit'+cls.__name__, self.default)
        return meth(node, **kw)

    def default(self, node, **kw):
        for child in node.getChildNodes():
            return self.visit(child, **kw)

    visitExpression = default
    
    def visitConst(self, node, **kw):
        return node.value

    def visitDict(self, node, **kw):
        return dict([(self.visit(k),self.visit(v)) for k,v in node.items])
        
    def visitTuple(self, node, **kw):
        return tuple(self.visit(i) for i in node.nodes)
        
    def visitList(self, node, **kw):
        return [self.visit(i) for i in node.nodes]

    def visitUnarySub(self, node, **kw):
        return -self.visit(node.getChildNodes()[0])
    
    def visitName(self, node, **kw):
        names = { 'true':True, 'false':False }
        name = node.name.lower()
        if name in names:
            return names[name]
        raise UnsafeSourceError("Unknown name", node.name, node)
    
class SafeEvalWithErrors(SafeEval):

    def default(self, node, **kw):
        raise Unsafe_Source_Error("Unsupported source construct",
                                node.__class__,node)
            
    # Add more specific errors if desired
            

def safe_eval(source, fail_on_error=True):
    walker = fail_on_error and SafeEvalWithErrors() or SafeEval()
    try:
        ast = compiler.parse(source,"eval")
    except SyntaxError, err:
        raise
    try:
        return walker.visit(ast)
    except UnsafeSourceError, err:
        raise


class JsonWriter:
    def dumps(self, v):
        """ Return the JSON-stringified version of the value `v`.
        """
        if isinstance(v, (bool, int, float)):
            return repr(v).lower()
        elif isinstance(v, long):
            return repr(v)[:-1] # drop the L suffix
        elif isinstance(v, (list, tuple)):
            return "[" + ",".join([ self.dumps(e) for e in v ]) + "]"
        elif isinstance(v, str):
            return '"' + v.replace('"', '\\"') + '"'
        elif isinstance(v, dict):
            return self.dumps_dict(v)
        else:
            raise Exception("Don't know how to serialize: %r" % v)

    def dumps_dict(self, v, comma=",", colon=":", first_keys=[]):
        """ Dump a dictionary to a JSON string.
            `comma` and `colon` override the the separators in the dictionary.
            `first_keys` is a list of keys to write first.
        """
        keys = v.keys()
        keys = [ fk for fk in first_keys if fk in keys ] + [ k for k in keys if k not in first_keys ]
        return "{" + comma.join([ self.dumps(k) + colon + self.dumps(v[k]) for k in keys ]) + "}"

class JsonReader:
    def loads(self, s):
        return safe_eval(s)

def loads(s):
    return safe_eval(s)

def dumps(v):
    return JsonWriter().dumps(v)
