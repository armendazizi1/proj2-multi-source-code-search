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
                firstLine=''
                if(comment != None):
                    firstLine = comment.splitlines()[0]
                    # print(node.name, '--', node.lineno, '==', 'class', "---", firstLine)
                df.loc[index_counter] = [node.name, fileName, node.lineno, 'class', firstLine]
                index_counter +=1


            for method in node.body:
                if(type(method) == ast.FunctionDef):
                    if (method.name[0] != "_" and method.name != "main" and method.name and "test" not in method.name.lower()):
                        # print(method.name, '--', method.lineno, '==', 'method')
                        comment = ast.get_docstring(method)

                        firstLine=''
                        if (comment != None):
                            firstLine = comment.splitlines()[0]
                            # print(method.name, '--', method.lineno, '==', 'class', "---", firstLine)
                        df.loc[index_counter] = [method.name, fileName, method.lineno, 'method', firstLine]
                        index_counter += 1


        elif(type(node) == ast.FunctionDef):
            if (node.name[0] != "_" and node.name != "main" and node.name and "test" not in node.name.lower()):
                # print(node.name, '--', node.lineno, '==', 'function', "---", firstLine)
                comment = ast.get_docstring(node)
                firstLine = ''
                if (comment != None):
                    firstLine = comment.splitlines()[0]
                    # print(node.name, '--', node.lineno, '==', 'function', "---", firstLine)
                df.loc[index_counter] = [node.name, fileName, node.lineno, 'function', firstLine]
                index_counter += 1

    return index_counter


def main():

    # path = '../myProj/'
    path = sys.argv[1]
    df = pd.DataFrame( columns=['name','file','line','type','comment'])
    # df.loc[0]=['a','b',23,'comment','type']
    # df.append(df2, ignore_index=True)

    index_counter=0
    print(df)
    for root, dirs, files in os.walk(path):
        for name in files:
            if name[-3:] == ".py":
                # print(name)
                fileName = os.path.join(root, name)

                shortPath = root+"/"+name
                file = open(fileName, 'r').read()
                astree = ast.parse(file);
                index_counter = visitTree(astree, index_counter, df, shortPath)
                # AstVisitor().visit(astree)
                # tree = javalang.parse.parse(str)
                # for path, node in tree.filter(javalang.tree.ClassDeclaration):
                #     class_names.append(tree.types[0].name)
                #     method_count.append(len(node.methods))
    print(df)
    df.to_csv('data' + ".csv")



if __name__ == "__main__":
    main()
