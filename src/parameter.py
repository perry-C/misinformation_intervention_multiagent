class AgentParameter:
    '''
    This just serves a little helper class for managing the 
    initialisation parameters of agents, overall greater 
    extensibility if I ever add more parameters in the future

    Also the reason for using a class rather than fixed values
    is for future proofing, in case I would like to keep things random
    '''

    def __init__(self, a, b, exp, flooding_capacity, inoculation_range):
        self.a = a
        self.b = b
        self.exp = exp
        self.flooding_capacity = flooding_capacity
        self.inoculation_range = inoculation_range
