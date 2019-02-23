import re

with open("setup_template.sh", "rt") as f:
    with open("setup.sh", "wt") as g:
        for line in f:
            m = re.match('(.*)export (.+)=""(.*)', line.strip())
            if m is not None:
                val = open(m.group(2), "rt").read().strip()
                g.write('{}export {}="{}"{}\n'.format(
                    m.group(1),
                    m.group(2),
                    val,
                    m.group(3),
                ))
            else:
                g.write(line)
