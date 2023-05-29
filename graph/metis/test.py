import mymetis

'''test graph
7 11 001
5 1 3 2 2 1
1 1 3 2 4 1
5 3 4 2 2 2 1 2
2 1 3 2 6 2 7 5
1 1 3 3 6 2
5 2 4 2 7 6
6 6 4 5
'''

# xadj, adjncy, vwgt, adjwgt, nparts | tpwgts, ufactor, dbg_lvl
xadj = [0,3,6,10,14,17,20,22]
adjncy = [
    5,3,2,
    1,3,4,
    5,4,2,1,
    2,3,6,7,
    1,3,6,
    5,4,7,
    6,4]
adjwgt = [
    1,2,1,
    1,2,1,
    3,2,2,2,
    1,2,2,5,
    1,3,2,
    2,2,6,
    6,5]
status = mymetis.partition(xadj=xadj, adjncy=adjncy, vwgt=None, adjwgt=adjwgt, nparts=2)
print(status)
