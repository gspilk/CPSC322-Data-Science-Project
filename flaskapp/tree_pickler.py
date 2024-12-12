import pickle

# Updated header to include TEAM and POSTSEASON
march_header = ["TEAM", "ADJOE", "ADJDE", "YEAR", "POSTSEASON"]

# Decision tree structure
march_tree = [
    "Attribute", "TEAM", [
        "Value", "Duke", [
            "Attribute", "ADJOE", [
                "Value", 125.2, [
                    "Attribute", "ADJDE", [
                        "Value", 90.6, [
                            "Attribute", "YEAR", [
                                "Value", 2015, ["Leaf", "Champions"]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    ],
    [
        "Value", "Other", ["Leaf", "Did not win the tournament"]
    ]
]

# Pack header and tree into a single object
packaged_obj = (march_header, march_tree)

# Save the tree to a pickle file
with open("flaskapp/tree.p", "wb") as outfile:
    pickle.dump(packaged_obj, outfile)
    outfile.close()

print("Decision tree saved to tree.p")
