import ast
import os
import sys
import pandas as pd


def visitTree(tree, counter, df, fileName):
    index_counter = counter
    for node in tree.body:
        if (type(node) == ast.ClassDef):
            if (node.name[0] != "_" and node.name != "main" and node.name and "test" not in node.name.lower()):
                comment = ast.get_docstring(node)
                firstLine = ''
                if (comment != None):
                    firstLine = comment.splitlines()[0]

                df.loc[index_counter] = [node.name, fileName, node.lineno, 'class', firstLine]
                index_counter += 1

            for method in node.body:
                if (type(method) == ast.FunctionDef):
                    if (method.name[
                        0] != "_" and method.name != "main" and method.name and "test" not in method.name.lower()):

                        comment = ast.get_docstring(method)

                        firstLine = ''
                        if (comment != None):
                            firstLine = comment.splitlines()[0]

                        df.loc[index_counter] = [method.name, fileName, method.lineno, 'method', firstLine]
                        index_counter += 1


        elif (type(node) == ast.FunctionDef):
            if (node.name[0] != "_" and node.name != "main" and node.name and "test" not in node.name.lower()):

                comment = ast.get_docstring(node)
                firstLine = ''
                if (comment != None):
                    firstLine = comment.splitlines()[0]

                df.loc[index_counter] = [node.name, fileName, node.lineno, 'function', firstLine]
                index_counter += 1

    return index_counter


def main():

    path = sys.argv[1]
    df = pd.DataFrame(columns=['name', 'file', 'line', 'type', 'comment'])
    index_counter = 0

    py_files_counter = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            if name[-3:] == ".py":

                py_files_counter += 1
                fileName = os.path.join(root, name)


                shortPath = root + name if root[-1:] == '/' else root + '/' + name
                shortPath = ".."+shortPath[shortPath.find('/tensorflow'):]

                file = open(fileName, 'r').read()
                astree = ast.parse(file);
                index_counter = visitTree(astree, index_counter, df, shortPath)

    df.to_csv('data' + ".csv")
    print("total number of python files: ", py_files_counter)


if __name__ == "__main__":
    main()
