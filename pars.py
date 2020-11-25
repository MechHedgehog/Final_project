import codecs

d = {}
i = 0
ex = ["invited", "January", "February", "March", "April", "May", "June", "July",
      "August", "September", "October", "November", "December",
      "In reply", "http", "Sticker", "Photo", "Not included", ".pdf", "pinned"]



def is_inic(l):
    global inpl
    if l == len(inpl) - 1:
        return 0
    bufа = ""
    if len(inpl[l+1].rstrip("\r\n")) == 5 and inpl[l+1][2] == ":":
        for i in inpl[l + 2].split(" ")[:len(inpl[l]) - 1]:
            bufа += i[0]
    else:
        for i in inpl[l + 1].split(" ")[:len(inpl[l]) - 1]:
            bufа += i[0]
    return (bufа[:len(inpl[l].rstrip("\r\n"))] == (inpl[l].rstrip("\r\n")))


def f(j):
    global inpl, ex
    buf = ""

    while j < len(inpl) and inpl[j] != "\r\n":
        k = 0
        while k != len(ex) and j < len(inpl):
            if inpl[j].find(ex[k]) != -1:
                j += 1
                if ex[k] == "Sticker" or ex[k] == "Photo":
                    j += 2
                if ex[k] == "Not included":
                    j += 1
                k = -1
            k += 1
        if j == len(inpl):
            return (buf, j - 1)
        if len(inpl[j].rstrip("\r\n")) == 5 and inpl[j][2] == ":":
            j += 1
            continue
        if is_inic(j):
            return (buf, j - 1)
        buf += inpl[j]
        j += 1

    return (buf, j)

x = ["mes.txt","mes2.txt","mes3.txt","mes4.txt","mes5.txt","mes6.txt",
     "mes7.txt","mes8.txt","mes9.txt","mes10.txt","mes11.txt","mes12.txt","mes13.txt","mes14.txt","mes15.txt","mes16.txt"] #3135135135135135
for q in x:
    inp = codecs.open(q, encoding='utf-8', mode='r')
    inpl = inp.readlines()
    inp.close()
    print(len(inpl))
    i = 0
    while i < len(inpl):
        if inpl[i] == "\r\n":
            i += 1
            continue
        if is_inic(i):
            if(len(inpl[i+1].rstrip("\r\n")) != 5 or len(inpl[i+1].rstrip("\r\n")) and inpl[i+1][2] != ":"):
                i += 1
                while i < len(inpl) and (not is_inic(i) or not (is_inic(i)
                    and not (len(inpl[i+1].rstrip("\r\n")) != 5 or len(inpl[i+1].rstrip("\r\n")) and inpl[i+1][2] != ":"))):
                    i += 1
            buf, j = f(i + 3)
            buf1 = ""
            for k in inpl[i + 2].split():
                if len(k) >= 6:
                    if k[2] == "." and k[5] == "." or k[2] == ":" and k[5] == ":":
                        continue
                buf1 += k + " "
            buf1 = buf1.rstrip("\n")
            buf = buf.rstrip("\r\n")
            buf += "\n"
            if not d.get(buf1):
                d[buf1] = ""
            if buf != "\n":
                d[buf1] += buf
            i = j
        i += 1



outp = codecs.open("outp.txt", encoding='utf-8', mode='w')
list_d = list(d.items())
list_d.sort(key=lambda i: len(i[1]), reverse=True)
for i in list_d:
    outp.write(i[0] + ":\n")
    outp.write(i[1])
    outp.write("############")
    outp.write("\n\n")




