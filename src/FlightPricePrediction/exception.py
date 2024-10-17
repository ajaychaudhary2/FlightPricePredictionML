import sys

class customexception(Exception):
    def __init__(self, errormessege, errordetails: sys):
        self.errormessege = errormessege
        # Correctly call sys.exc_info() to get the traceback
        _, _, exc_tb = sys.exc_info()
        
        self.line_n = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename
        
    def __str__(self):
        return "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(
            self.filename, self.line_n, str(self.errormessege)
        )        

 
