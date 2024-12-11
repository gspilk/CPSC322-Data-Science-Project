import pickle

march_header = ["TEAM", "ADJOE", "ADJDE", "EFF_O", "EFF_D", "YEAR"]  # The header representing feature names

march_tree = [
    'Attribute', 'TEAM', 
    ['Value', 'Duke', 
        ['Attribute', 'ADJOE', 
            ['Value', '125.2', ['Leaf', 'True', 1, 3523]], 
            ['Value', 'no', ['Leaf', 'False', 3522, 3523]]
        ],
        ['Attribute', 'ADJDE', 
            ['Value', '90.6', ['Leaf', 'True', 1, 3523]], 
            ['Value', 'no', ['Leaf', 'False', 3522, 3523]]
        ],
        ['Attribute', 'EFF_O', 
            ['Value', '0.9764', ['Leaf', 'True', 1, 3523]], 
            ['Value', 'no', ['Leaf', 'False', 3522, 3523]]
        ],
        ['Attribute', 'EFF_D', 
            ['Value', '56.6', ['Leaf', 'True', 1, 3523]], 
            ['Value', 'no', ['Leaf', 'False', 3522, 3523]]
        ],
        ['Attribute', 'YEAR', 
            ['Value', '2015', ['Leaf', 'True', 1, 3523]], 
            ['Value', 'no', ['Leaf', 'False', 3522, 3523]]
        ]
    ]
]

# Pickle (serialize) header and tree together
packaged_obj = (march_header, march_tree)
outfile = open("flaskapp/tree.p", "wb")
pickle.dump(packaged_obj, outfile)
outfile.close()
