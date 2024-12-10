import pickle # standard library

march_header = ["TEAM", "ADJOE", "ADJDE", "EFF_O", "EFF_D", "YEAR"] # TEAM, OFFADJ, DEFADJ, EFF_O, EFF_D
march_tree = ['Attribute', 'TEAM', 
              ['Value', 'Duke', 
               ['Attribute', 'ADJOE', 
                ['Value', '125.2', ['Leaf', 'True', 1, 3523]], 
                ['Value', 'no', ['Leaf', 'False', 3522, 3523]]]
              ]
]

# pickle (object serialization): saving a binary representation of an object
# to file for loading and using later
# example: saving a trained a model for inference/prediction later
# in another python process, possibly running on a diff machine (server)
# imagine you just trained an awesome MyRandomForestClassifier
# and now you need to save it for using in your web app on a server later
# de/unpickle (object deserialization): loading a binary representation of
# an object from a file into a python object in program memory
# example: a web app that loads the trained model up for inference/prediction
# requests from clients

# lets pickle header and tree (together)
packaged_obj = (march_header, march_tree)
outfile = open("tree.p", "wb")
pickle.dump(packaged_obj, outfile)
outfile.close()