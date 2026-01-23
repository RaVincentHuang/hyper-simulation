import re

record = '("entity", "Guglielmo Caccia", "Person", "An artist mentioned in the list.");'
# record = '"entity", "School of Giovanni Battista", "Attribute", "Indicates a work created by the school or workshop associated with Giovanni Battista."'

def _out_bars(record):
    match1 = re.match(r'\((.*)\);', record)
    match2 = re.match(r'\((.*)\)', record)
    if match1:
        res: str = match1.group(1)
        return res
    elif match2:
        res: str = match2.group(1)
        return res
    else:
        return record
    

match = re.match(r'"entity",(.+),(.+),(.+)', _out_bars(record))
print(match)
if match:
    name, type_, desc = match.groups()
    print(name, type_, desc)
