from __future__ import print_function
import json
import pandas
import scipy.sparse as sp
import numpy as np
import sys
import math
import logging
import itertools
from mpi4py.MPI import DOUBLE
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc as petsc

def recommend(userid, user_items, user_factors, item_factors, N=10):
    # user = self._user_factor(userid, user_items, recalculate_user)
    users, items = user_items.getSize()
    _, factors = user_factors.getSize()

    user = petsc.Vec()
    user.createSeq(factors)
    user.setValues(list(range(factors)), user_factors.getValues([userid], list(range(factors))))
    user.assemble()
    #user.view()
    # # calculate the top N items, removing the users own liked items from the results
    # liked = set(user_items[userid].indices)
    liked = user_items.getRow(userid)[0]
    # print(liked)
    # scores = self.item_factors.dot(user)
    scores = petsc.Vec()
    scores.createSeq(items)
    scores.assemble()

    item_factors.mult(user, scores)
    #scores.view()

    scores_np = scores.getArray()
    #print(scores)
    # borrowed from implicit
    count = N + len(liked)

    scores_np = zip(range(0,len(scores_np)), scores_np)
    scores_np = [i for i in scores_np if i[0] not in liked]
    scores_np = sorted(scores_np, key=lambda x: -x[1])


    result = scores_np[:N]
    # print(result)

    # if count < len(scores_np):
    #     ids = np.argpartition(scores_np, -count)[-count:]
    #     best = sorted(zip(ids, scores_np[ids]), key=lambda x: -x[1])
    # else:
    #     best = sorted(enumerate(scores_np), key=lambda x: -x[1])
    #
    # result = list(itertools.islice((rec for rec in best if rec[0] not in liked), N))

    user.destroy()
    del(liked)
    scores.destroy()
    del(scores_np)
    # del(ids)
    # del(best)

    return result

comm = petsc.COMM_WORLD.tompi4py()
rank = comm.Get_rank()
num_procs = comm.Get_size()

print("Hello, I'm %d of %d" % (rank, num_procs))

print("reading things into pandas")
sys.stdout.flush()

indptr  = []
indices = []
data    = []
with open("tropes_01-22-2018.csr", "r") as infile:
    for line in infile:
        if len(line.split(",")) == 1:
            indptr.append(len(indices))
        else:
            col, rating = line.split(",")
            indices.append(int(col))
            data.append(float(rating))

mat = sp.csr_matrix((data, indices, indptr))

# free the memory
del(indptr)
del(indices)
del(data)

print(mat.shape)

items, users = mat.shape
regularization = 0.01 #default borrowed from implicit
factors = 350

sys.stdout.flush()

print("item users\n")
item_users = petsc.Mat()
item_users.create(petsc.COMM_SELF)
item_users.setSizes(mat.shape)
item_users.setType("seqaij")
item_users.setUp()

item_users.setPreallocationCSR((mat.indptr, mat.indices))
item_users.setValuesCSR(mat.indptr, mat.indices, mat.data)

item_users.assemble()

sys.stdout.flush()

print("transpose")
matT = mat.T.tocsr()

del(mat)
#del(indptr)
#del(indices)
#del(data)
print("user items")
user_items = petsc.Mat()
user_items.create(petsc.COMM_SELF)
user_items.setSizes(matT.shape)
user_items.setType("seqaij")
user_items.setUp()

user_items.setPreallocationCSR((matT.indptr, matT.indices))
user_items.setValuesCSR(matT.indptr, matT.indices, matT.data)

user_items.assemble()

wf = []
#TODO: read in matrices
with open("userfactors.txt", "r") as uf:
    for line in uf:
        l = line.split(" ")
        wf.append(list(map(float, l)))
tf = []
#TODO: read in matrices
with open("itemfactors.txt", "r") as uf:
    for line in uf:
        l = line.split(" ")
        tf.append(list(map(float, l)))

print("work factors")
user_factors = petsc.Mat()
user_factors.create(petsc.COMM_SELF)
user_factors.setSizes((users, factors))
user_factors.setType("seqdense")
user_factors.setUp()

user_factors.setValues(list(range(0,users)), list(range(factors)), wf[0:users])

user_factors.assemble()

print(user_factors.getOwnershipRange())

#user_factors.view()
print("trope factors")
item_factors = petsc.Mat()
item_factors.create(petsc.COMM_SELF)
item_factors.setSizes((items, factors))
item_factors.setType("seqdense")
item_factors.setUp()

item_factors.setValues(list(range(0,items)), list(range(factors)), tf[0:items])

item_factors.assemble()


wl = ["RWBY", "FullmetalAlchemist2017", "GunnerkriggCourt",
        "AttackOnTitan", "TheLastJedi", "TheElderScrollsIIIMorrowind",
        "TheDresdenFiles", "Hamilton"]

id_trope = json.load(open("id_to_trope.json","r"))
id_trope = dict((int(k), v) for k,v in id_trope.items())
#trope_id = dict((v,k) for k,v in id_trope.items())

# t = trope_id["SmartPeoplePlayChess"]

#print(tropes)
# for other_trope, score in model.similar_items(t, 10):
#     print(id_trope[other_trope] + ", " + str(score))

id_work = json.load(open("id_to_work.json", "r"))
work_id = dict((v,k) for k,v in id_work.items())


if petsc.COMM_WORLD.Get_rank() == 0:
    for name in wl:
        w = int(work_id[name])
        print()
        print(w)
        print(petsc.COMM_WORLD.Get_rank())
        print(name)
        for t_id, score in recommend(w, user_items, user_factors, item_factors):
            print(id_trope[t_id] + ", " + str(score))

sys.exit(0)
