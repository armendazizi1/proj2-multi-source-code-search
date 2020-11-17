import ast
import os
import sys
import pandas as pd



def visitTree(tree, counter, df, fileName):
    index_counter = counter
    for node in tree.body:
        if(type(node) == ast.ClassDef or type(node) == ast.FunctionDef):
            if (node.name[0] != "_" and node.name != "main" and node.name and not "test" in node.name  ):
                if (type(node) == ast.ClassDef):

                    comment = ast.get_docstring(node)
                    if(comment != None):
                        firstLine = comment.splitlines()[0]
                        # print(node.name, '--', node.lineno, '==', 'class', "---", firstLine)
                        df.loc[index_counter] = [node.name, fileName, node.lineno, 'class', firstLine]
                        index_counter +=1


                    for method in node.body:
                        if(type(method) == ast.FunctionDef):
                            # print(method.name, '--', method.lineno, '==', 'method')
                            comment = ast.get_docstring(method)
                            if (comment != None):
                                firstLine = comment.splitlines()[0]
                                # print(method.name, '--', method.lineno, '==', 'class', "---", firstLine)
                                df.loc[index_counter] = [method.name, fileName, method.lineno, 'method', firstLine]
                                index_counter += 1


                elif(type(node) == ast.FunctionDef):
                    # print(node.name, '--', node.lineno, '==', 'function', "---", firstLine)
                    comment = ast.get_docstring(node)
                    if (comment != None):
                        firstLine = comment.splitlines()[0]
                        # print(node.name, '--', node.lineno, '==', 'function', "---", firstLine)
                        df.loc[index_counter] = [node.name, fileName, node.lineno, 'function', firstLine]
                        index_counter += 1

    return index_counter


def main():

    path = '../tensorflow/'
    path = sys.argv[1]
    df = pd.DataFrame( columns=['name','file','line','type','comment'])
    # df.loc[0]=['a','b',23,'comment','type']
    # df.append(df2, ignore_index=True)

    index_counter=0
    # print(df)
    for root, dirs, files in os.walk(path):
        for name in files:
            if name[-3:] == ".py":
                # print(name)
                fileName = os.path.join(root, name)
                file = open(fileName, 'r').read()
                astree = ast.parse(file);
                index_counter = visitTree(astree, index_counter, df, fileName)
                # AstVisitor().visit(astree)
                # tree = javalang.parse.parse(str)
                # for path, node in tree.filter(javalang.tree.ClassDeclaration):
                #     class_names.append(tree.types[0].name)
                #     method_count.append(len(node.methods))
    # print(df)
    df.to_csv('data' + ".csv")



if __name__ == "__main__":
    main()
