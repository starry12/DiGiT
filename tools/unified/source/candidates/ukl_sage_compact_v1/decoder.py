"""Bound decoder launch description; actual process deployment remains gated."""
from .common import HERE,require,heavy_gate
from .stream import validate_header


def plan():
    return dict(source=str(HERE/'decoder/UKLStream.java'),webgraph_version='3.6.12',jdk='11 or compatible',
        compiled=False,execute=False,full_graph_decode_accepted=False,
        required=['JDK and pinned WebGraph transitive jars with hashes','verified incoming/transpose source and node-map identity',
                  'compiled class receipt','resource budgets and original grid completion'],
        stream_stdout='DIGUKL01, bound JSON, owner i32/degree i64/neighbor i32 big endian, DIGEND01/edge count',
        memory_note='Python reader <=65536 neighbors/chunk; WebGraph iterator internals may retain a successor/reference window; budget separately')


def launch_command(binding,basename,header_path,classpath,heap_mib=1024):
    # No launch in this function. A future controller binds artifacts, owns locks,
    # writes the header, supervises exit/footer and supplies resource admission.
    validate_header(binding)
    require(binding['orientation']=='incoming','Untransposed source cannot feed CSC')
    require(type(heap_mib) is int and 256<=heap_mib<=8192,'Invalid planned Java heap')
    return ['java','-Xmx%dm'%heap_mib,'-cp',str(classpath),'UKLStream',str(basename),str(header_path),
            str(binding['nodes']),str(binding['edges'])]


def execute():
    heavy_gate()
    raise ValueError('Source-only decoder: pin jars/class/source receipts and implement supervised deployment first')
