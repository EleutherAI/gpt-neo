with open("/home/connor/sid/fix_inputs/GPTNeo/data/test.txt", "w") as f:
    s = ", ".join(str(i) for i in list(range(0, 4000)))
    print(s)
    f.write(s)

with open("/home/connor/sid/fix_inputs/GPTNeo/data/test2.txt", "w") as f:
    s = ", ".join(str(i) for i in list(range(4000, 8000)))
    print(s)
    f.write(s)

with open("/home/connor/sid/fix_inputs/GPTNeo/data/test3.txt", "w") as f:
    s = ", ".join(str(i) for i in list(range(8000, 12000)))
    print(s)
    f.write(s)

with open("/home/connor/sid/fix_inputs/GPTNeo/data/test4.txt", "w") as f:
    s = ", ".join(str(i) for i in list(range(12000, 14000)))
    print(s)
    f.write(s)