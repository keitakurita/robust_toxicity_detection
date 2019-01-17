import re

with open("setup_template.sh", "rt") as f:
    with open("setup.sh", "wt") as g:
        for line in f:
            m = re.match('export (.+)=""', line.strip())
            if m is not None:
                val = open(m.group(1), "rt").read().strip()
                g.write('export {}="{}"\n'.format(m.group(1),
                                               val))
            else:
                g.write(line)
