import argparse
import codecs
import os
import time

from pathos.pools import ProcessPool
from pathos.helpers import cpu_count

class MDAParser(object):
    def __init__(self):
        pass

    def extract(self, txt_dir, mda_dir, parsing_log):
        self.txt_dir = txt_dir
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        self.mda_dir = mda_dir
        if not os.path.exists(mda_dir):
            os.makedirs(mda_dir)

        def text_gen(txt_dir):
            # Yields markup & name
            for fname in os.listdir(txt_dir):
                if not fname.endswith('.txt'):
                    continue
                yield fname

        def parsing_job(fname):
            print("Parsing: {}".format(fname))

            # Read text
            filepath = os.path.join(self.txt_dir,fname)
            with codecs.open(filepath,'rb',encoding='utf-8') as fin:
                text = fin.read()

            name, ext = os.path.splitext(fname)
            # Parse MDA part

            msg = ""
            mda, end = self.parse_mda(text)
            # Parse second time if first parse results in index
            if mda and len(mda.encode('utf-8')) < 500:
                mda, _ = self.parse_mda(text, start=end)

            if mda: # Has value
                msg = "SUCCESS"
                mda_path = os.path.join(self.mda_dir, name + '.mda')
                with codecs.open(mda_path,'w', encoding='utf-8') as fout:
                    fout.write(mda)
            else:
                msg = msg if mda else "MDA NOT FOUND"
            #print("{},{}".format(name,msg))
            return name + '.txt', msg #


        ncpus = cpu_count() if cpu_count() <= 8 else 8
        pool = ProcessPool( ncpus )

        _start = time.time()
        parsing_failed = pool.map( parsing_job, \
                                   text_gen(self.txt_dir) )
        _end = time.time()

        print("MDA parsing time taken: {} seconds.".format(_end-_start))

        # Write failed parsing list
        count = 0
        with open(parsing_log,'w') as fout:
            print("Writing parsing results to {}".format(parsing_log))
            for name, msg in parsing_failed:
                fout.write('{},{}\n'.format(name,msg))
                if msg != "SUCCESS":
                    count = count + 1

        print("Number of failed text:{}".format(count))

    def parse_mda(self, text, start=0):
        debug = False
        """
            Return Values
        """

        mda = ""
        end = 0

        """
            Parsing Rules
        """

        # Define start & end signal for parsing
        item2_begins = ['MANAGEMENT\'S DISCUSSION', 'Management\'s Discussion','MANAGEMENT\'S  DISCUSSION','Results of Operations', 'RESULTS OF OPERATIONS','Item 2','ITEM 2','PLAN OF OPERATIONS','PLAN OF OPERATION','Royalty  Revenues']
        #[ '\nITEM 2.', '\nItem 2.','\nITEM 2 –','\nItem 2','\nITEM 2:', '\nITEM 2 ', '\nITEM 2\n', "\nMANAGEMENT'S DISCUSSION " ]
        item2_ends   = ['PART II','Part II','OTHER INFORMATION','Other Information','LEGAL PROCEEDINGS','Item 6','ITEM 6','Legal Proceedings','SIGNATURES','Pursuant to the requirements','In accordance with the requirements','EXHIBIT 11']
        #[ '\nPART II ' ,'\nPART II.','\nPART II - ']
        #if start != 0:
            #item2_ends.append('\nITEM 2') # Case: ITEM 7A does not exist
        #part2_begins = [ '\nPART II ' ,'\nPART II. ','\nPART II - ' ]

        """
            Parsing code section
        """
        text = text[start:]

        # Get begin
        for item2 in item2_begins:
            begin = text.find(item2)
            if debug:
                print(item2,begin)
            if begin != -1:
                break
                
        if begin != -1: # Begin found
            for part2 in item2_ends:
                end = text.find(part2, begin+1)
                if debug:
                    print(part2,end)
                if end != -1:
                    break

            #if end == -1: # ITEM 7A does not exist
                #for part2 in part2_begins:
                    #end = text.find(part2, begin+1)
                    #if debug:
                        #print(part2,end)
                    #if end != -1:
                        #break

            # Get MDA
            if end > begin:
                mda = text[begin:end].strip()
            else:
                end = 0

        return mda, end

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse MDA section of Edgar Form 10k")
    parser.add_argument('--txt_dir',type=str,default='./data/txt')
    parser.add_argument('--mda_dir',type=str,default='./data/mda')
    parser.add_argument('--log_file',type=str,default='./parsing.log')
    args = parser.parse_args()

    # Extract MD&A from processed text
    # Note that the parser parses every text in the text_dir, not according to the index file
    parser = MDAParser()
    parser.extract(txt_dir=args.txt_dir, mda_dir=args.mda_dir, parsing_log=args.log_file)
