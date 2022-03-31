def getLatE(element):
    """
    read the Energy of single crystal 
    """
    with open(f'C:/Users/hatay/calc/python_script/{element}/atomsout','r') as f:
        lines=f.readlines()
        LatE = lines[-4:][0].split()[4].replace(';', '')
    return float(LatE)

def getLatP(element):
    """
    read the Lattice Parameter
    """
    with open(f'C:/Users/hatay/calc/python_script/{element}/atomsout','r') as f:
        lines=f.readlines()
        LatP = lines[-5:][0].split()[4].replace(';', '')
    return float(LatP)

def get_mass(element):
    with open(f'C:/Users/hatay/calc/python_script/{element}/mass','r') as f:
        lines=f.read()
    return float(lines)


class Element:
    def __init__(self,element):
        self.element = element
        
    def grand(self):
        return getLatE(self.element)
    
    def getLatP(self):
        return getLatP(self.element)

    def mass(self):
        return get_mass(self.element)
        

    def meam(self):
        with open(f"C:/Users/hatay/calc/python_script/{self.element}/{self.element}.meam","r") as f:
            lines = f.readlines()
        return lines

    def library_meam(self):
        with open(f"C:/Users/hatay/calc/python_script/{self.element}/library.meam","r") as f:
            lines = f.readlines()
        return lines
    def proto(self):
        with open(f"C:/Users/hatay/calc/python_script/{self.element}/proto.in","r") as f:
            lines = f.readlines()
            return lines
