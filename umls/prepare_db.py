import umls_lib
import collections
import os
import sqlite3

# cat META/MRCONSO.RRF | tr '|' '\t' | cut -f 1,8,12,13
# cat META/MRSTY.RRF | tr '|' '\t' | cut -f1,2
# cat META/MRREL.RRF | tr '|' '\t' | cut -f1,4,5,8,11

Table = collections.namedtuple('Table',
'name origin_file field_indices field_names primary_key')

atoms = Table('atoms', umls_lib.CONCEPT_FILE,
    [0, 7, 11], ["cui", "aui", "source"], "aui")
relations = Table('relations', umls_lib.RELATION_FILE,
    [0, 3, 4, 7, 10], [], "(cui1, cui2)")

def construct_table(table, conn):
  c = conn.cursor()
  create_table_string = "CREATE TABLE {} ({}, PRIMARY KEY ({}))".format(
      table.name,
      ", ".join(name+" TEXT" for name in table.field_names),
      table.primary_key)
  print(create_table_string)
  c.execute(create_table_string)

  file_path = os.path.join("/iesl/data/umls_2017AB", table.origin_file)
  for fields in umls_lib.read_file(file_path):
    values = {field_name:fields[field_index]
        for (field_name, field_index)
        in zip( table.field_names, table.field_indices)}
    insert_string = "INSERT INTO {} VALUES ({})".format(
        table.name,
        ", ".join('"' + values[name] + '"' for name in table.field_names))
    print(insert_string)
    c.execute(insert_string)
  conn.commit()

def main():
  conn = sqlite3.connect(umls_lib.DB_FILE)
  construct_table(atoms, conn)
  construct_table(relations, conn)
  construct_table(types, conn)

  conn.close()


if __name__=="__main__":
  main()
