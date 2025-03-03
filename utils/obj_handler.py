import numpy as np

def open_obj(file_name):
    vertices, faces = [], []

    with open(file_name) as f:
        for l in f:
            tokens = l.split()

            if not tokens:
                continue

            # vertice processing
            if tokens[0] == "v":
                vertices.append(tokens[1:4] + [1,1,1]) # add some vector for projection

            if tokens[0] == "f":
                faces.append([t.split('/')[0] for t in tokens[1:4]])
                
                # add a second triangle
                # Triangulate?
                if len(tokens) > 4:
                    faces.append([t.split('/')[0] for t in (tokens[1], tokens[3], tokens[4])])

    return np.array(vertices).astype(float), (np.array(faces).astype(int)-1)
