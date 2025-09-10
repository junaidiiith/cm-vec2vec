from typing import Union, Literal


def get_resources(node):
    
    def get_resources_util(n):
        if 'resourceId' in n:
            resources[n['resourceId']] = n
        if 'childShapes' in n and isinstance(n['childShapes'], list):
            for child in n['childShapes']:
                get_resources_util(child)
    
    resources = dict()
    get_resources_util(node)
    return resources


def check_resource_presence(model, resources):    
    def check_resource_util(node):
        if 'outgoing' in model and len(model['outgoing']):
            for outgoing in model['outgoing']:
                assert outgoing['resourceId'] in resources, f"{outgoing['resourceId']} not in resources"
        
        if 'childShapes' in node and isinstance(model['childShapes'], list):
            for child in node['childShapes']:
                check_resource_util(child)
                
    check_resource_util(model)


def get_node_name(**kwargs):
    properties = kwargs.get('properties', {})
    level = kwargs.get('level')
    name = properties.get('name', '')
    description = f"{'\t'*(level)} is described by: {properties.get('documentation', '')}" if properties.get('documentation', '') else ""
    return f"{name}{description}"


def get_node_str(**kwargs):
    level = kwargs.get('level')
    node_name = get_node_name(**kwargs)
    node_type = kwargs.get('stencil') if 'stencil' in kwargs and 'id' in kwargs.get('stencil') else ''

    node_str = f"{'\t'*(level)}Node: ({node_type}) {node_name}" if node_name else ""
    return node_str
    
    
def get_edges_str(**kwargs):
    resource_map = kwargs.get('resource_map')
    level = kwargs.get('level')
    edges = kwargs.get('outgoing', [])
    edges_str= ""
    edges_content = [
        get_node_name(level=level, **resource_map[r['resourceId']]) 
        for r in edges if r['resourceId'] in resource_map
    ]
    edges_content = [e for e in edges_content if len(e)]
    if len(edges_content):
        edges_str = f"\n{'\t'*(level)}".join(edges_content)   
    return edges_str


def serialize_bpmn_nl_template(**kwargs):
    node_str = get_node_name(**kwargs)
    return node_str


def serialize_bpmn_cm_template(**kwargs):
    level = kwargs.get('level')
    node_str = get_node_str(**kwargs)
    edges_str = get_edges_str(**kwargs)
    return f"{node_str}\n{'\t'*(level)}Connections:\n {edges_str}".strip()


def serialize_model(model, resources, stype, level=0, use_structure=True):
    serializer = serialize_bpmn_cm_template if stype == 'cm' else serialize_bpmn_nl_template
    node_data = serializer(resource_map=resources, level=level, use_structure=use_structure, **model)
    child_data = list()
    if 'childShapes' in model and isinstance(model['childShapes'], list):
        for child in model['childShapes']:
            child_str = serialize_model(child, resources=resources, stype=stype, level=level+1)
            if child_str:
                child_data.append(child_str)
    
    if stype == 'nl':
        data_str = f"{node_data}" + f"\n".join(child_data)
    else:
        data_str = f"{'\t'*level}{node_data}\n" + f"{'\t'*(level+1)}\n".join(child_data)
    return data_str


def serialize_bpmn_model(model, stype=Union[Literal['cm', 'nl']], level=0, use_structure=True):
    resources = get_resources(model)
    check_resource_presence(model, resources)
    return serialize_model(model, resources=resources, stype=stype, level=level, use_structure=use_structure)
