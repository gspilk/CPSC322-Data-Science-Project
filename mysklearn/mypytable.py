import copy
import csv
import os
from tabulate import tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        rows = 0 
        col = 0
        col = len(self.column_names)
        for row in self.data:
            rows += 1
        return rows, col #finished function

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_table = []
        index = 0
        if isinstance(col_identifier, str):
            index = self.find_col_num(self.column_names, col_identifier)
            if index == -1:
                raise ValueError("Invalid col_identifier used")
        elif isinstance(col_identifier, int):
            index = col_identifier
        else:
            raise ValueError("Invalid col_identifier used")

        if(include_missing_values == False):
            self.remove_rows_with_missing_values()
        for i in range(len(self.data)):
            col_table.append(self.data[i][index])
        return col_table # TODO: fix this

    def find_col_num(self, header, col_name):
        """Created a helper function to find the index of which column is selected.

            Args:
                header (list of str): Column names corresponding to the table
                col_name (str): Represents the name of the column to check for missing values
            Returns: 
                Integer: The index of the column
        """
        i = 0
        while i < len(header):
            #print(header[i])
            if(header[i] == col_name):
                return i
            i += 1
        return -1

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        error_counter = 0 #throw away variable so that the except will work
        for i in range(len(self.data)):
            for j in range(len(self.column_names)):
                try:
                    self.data[i][j] = float(self.data[i][j])
                    #print(float(self.data[i][j]))
                except ValueError:
                    #print(f"Invalid Conversion of value: {self.data[i][j]}")
                    error_counter += 1
                    
            
        pass # TODO: fix this

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """

        for num in sorted(row_indexes_to_drop, reverse=True):
            try:
                #print(num)
                del self.data[num]
            except IndexError:
                print(f"Index {num} out of range")

                

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        file = open(filename, "r")
        reader = csv.reader(file)
        self.data = list(reader)
        self.column_names = self.data[0]
        self.data.remove(self.column_names)
        self.convert_to_numeric()

        file.close()
        return self #finished

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        file = open(filename, "w")
        for i in range(len(self.column_names)):
            file.write(str(self.column_names[i]))
            if i < len(self.column_names) - 1:
                file.write(",")
        file.write("\n")

        for i in range(len(self.data)):
            for j in range(len(self.column_names)):
                file.write(str(self.data[i][j]))
                if j < len(self.column_names) - 1:
                    file.write(",")
            file.write("\n")
        file.close()   
        pass #Finished

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str or int): column names or column indices to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        col_indexes = []
        
        for col_identifier in key_column_names:
            if isinstance(col_identifier, str):
                index = self.find_col_num(self.column_names, col_identifier)
                if index == -1:
                    raise ValueError(f"Invalid column name: {col_identifier}")
            elif isinstance(col_identifier, int):
                index = col_identifier
            else:
                raise ValueError("Invalid col_identifier used")
            col_indexes.append(index)

        seen_rows = {}
        dup_rows = []

        for i, row in enumerate(self.data):
            row_key = tuple(row[idx] for idx in col_indexes)
            
            if row_key in seen_rows:
                dup_rows.append(i)  
            else:
                seen_rows[row_key] = i  
        return dup_rows #finished



    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        i = 0
        while i < len(self.data):
            if "NA" in self.data[i]:
                del self.data[i]
            else:
                i += 1
        pass

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.find_col_num(self.column_names, col_name)
        if col_index == -1:
            raise ValueError(f"Column '{col_name}' does not exist")
        
        total = 0
        count = 0
        for row in self.data:
            value = row[col_index]
            if value != "NA":
                total += float(value)
                count += 1
        
        column_average = total / count

        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = column_average

        pass #finished

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        if len(self.data) == 0:
            return MyPyTable()
        
        col_indexes = []
        
        for col_identifier in col_names:
            if isinstance(col_identifier, str):
                index = self.find_col_num(self.column_names, col_identifier)
                if index == -1:
                    raise ValueError(f"Invalid column name: {col_identifier}")
            elif isinstance(col_identifier, int):
                index = col_identifier
            else:
                raise ValueError("Invalid col_identifier used")
            col_indexes.append(index)

        stats_table = []
        for index in col_indexes:
            # Extract the current column and filter out non-numeric values
            curr_table = [row[index] for row in self.data if row[index] not in ("NA", "")]
            # Convert to float
            curr_table = [float(value) for value in curr_table if isinstance(value, (int, float, str)) and str(value).replace('.', '', 1).isdigit()]

            if not curr_table:
                # Handle the case where the column has no valid numeric entries
                new_table = [self.column_names[index], "NA", "NA", "NA", "NA", "NA"]
            else:
                new_table = [
                    self.column_names[index],
                    min(curr_table),
                    max(curr_table),
                    self.find_mid(curr_table),
                    self.find_avg(curr_table),
                    self.find_median(curr_table)
                ]
            stats_table.append(new_table)
        
        new_mypytable = MyPyTable(column_names=["attribute", "min", "max", "mid", "avg", "median"], data=stats_table)
        return new_mypytable


    def find_mid(self, table):
        """Finds the middle value of a list

        Args:
            table(list): list full of data

        Returns: 
            float: mid value of the list        
        """
        return (min(table) + max(table)) / 2
    
    def find_avg(self, table):
        """Finds the average of a list

        Args:
            table(list): list full of data

        Returns: 
            float: average value of the list        
        """
        total = 0
        count = 0
        for num in table:
            total += float(num)
            count += 1
        
        column_average = total / count
        return column_average
    
    def find_median(self, table):
        """Finds the median value of the list
        
        Args: table(list): list full of data

        Returns:
            float: median value of the lsit

        """
        sorted_numbers = sorted(table)
        length = len(sorted_numbers)

        if length % 2 == 1:  
            return sorted_numbers[length // 2]
        else:  
            middle1 = sorted_numbers[length // 2 - 1]
            middle2 = sorted_numbers[length // 2]
            return (middle1 + middle2) / 2        
    
    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        key_indices_self = [self.column_names.index(col) for col in key_column_names]
        key_indices_other = [other_table.column_names.index(col) for col in key_column_names]
        joined_column_names = self.column_names + [
            col for col in other_table.column_names if col not in key_column_names
        ]
    
        joined_table = []
        for row1 in self.data:
            for row2 in other_table.data:
                if all(row1[key_indices_self[i]] == row2[key_indices_other[i]] for i in range(len(key_column_names))):
                    joined_row = row1 + [row2[other_table.column_names.index(col)] for col in other_table.column_names if col not in key_column_names]
                    joined_table.append(joined_row)
        
        return MyPyTable(joined_column_names, joined_table) # TODO: fix this


    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        
        key_indices_self = [self.column_names.index(col) for col in key_column_names]
        key_indices_other = [other_table.column_names.index(col) for col in key_column_names]
        joined_column_names = self.column_names + [
            col for col in other_table.column_names if col not in key_column_names
        ]

        matched_rows_self = set()
        matched_rows_other = set()
        joined_data = []

        for i, row1 in enumerate(self.data):
            match_found = False
            for j, row2 in enumerate(other_table.data):
                if all(row1[key_indices_self[k]] == row2[key_indices_other[k]] for k in range(len(key_column_names))):
                    merged_row = row1 + [row2[other_table.column_names.index(col)] for col in other_table.column_names if col not in key_column_names]
                    joined_data.append(merged_row)
                    match_found = True
                    matched_rows_self.add(i)
                    matched_rows_other.add(j)
            if not match_found:
                joined_data.append(row1 + ["NA"] * (len(other_table.column_names) - len(key_column_names)))

        
        for j, row2 in enumerate(other_table.data):
            if j not in matched_rows_other:
                padded_row = ["NA"] * len(self.column_names)  
                for col in key_column_names:
                    padded_row[self.column_names.index(col)] = row2[other_table.column_names.index(col)]  
                padded_row += [row2[other_table.column_names.index(col)] for col in other_table.column_names if col not in key_column_names]
                joined_data.append(padded_row)

        return MyPyTable(joined_column_names, joined_data)