package polya;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import java.sql.Timestamp;
import java.util.Date;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;

/**
 *
 * class for preprocessing the old med, cisi and cran test collections.
 * 
 */
public class Preprocess {
    
    
        public static void preprocess(String filename) throws FileNotFoundException, IOException {

        Analyzer analyzer  = new EnglishAnalyzer();
        
        StringBuilder strb = null;
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line, toks[];
            String id = "";
            while ((line = br.readLine()) != null) {

                if (line.startsWith(".I")) {
                    if (strb!=null) System.out.println(id + "\t" +stem(strb.toString(),analyzer));
                    strb = new StringBuilder();
                    id = line.trim();
                    continue;
                }
                
                if (line.startsWith(".W")||line.startsWith(".T")||
                        line.startsWith(".X")||line.startsWith(".B")||line.startsWith(".A")) {
                    continue;
                }

                if (strb != null) {
                    line = line.trim();
                    toks = line.split(" ");
                    for (String t : toks){
                        strb.append(t).append(" ");
                    }
                    //strb.append(line.replace("\n", " ")).append(" ");
                }
            }
            System.out.println(id + "\t" + stem(strb.toString(),analyzer));
        }
    }
        
        
    public static String stem(String text, Analyzer analyzer) throws IOException{

        TokenStream ts = analyzer.tokenStream("myfield", new StringReader(text));
        OffsetAttribute offsetAtt = ts.addAttribute(OffsetAttribute.class);
        StringBuilder processed_text = new StringBuilder();
        try {
            ts.reset(); // Resets this stream to the beginning. (Required)
            while (ts.incrementToken()) {
                //index terms greater than length 2
                if ((ts.getAttribute(CharTermAttribute.class).toString().length() > 2)&&
                        !(ts.getAttribute(CharTermAttribute.class).toString().matches(".*\\d.*"))){
                    processed_text.append(ts.getAttribute(CharTermAttribute.class).toString()).append(" ");
                }
            }
            ts.end();   // Perform end-of-stream operations, e.g. set the final offset.
        } finally {
            ts.close(); // Release resources associated with this stream.
        }
        return processed_text.toString();
    }

    public static void main(String[] args) throws IOException {
        

        System.err.println(new Timestamp(new Date().getTime()));
        if (args.length == 1) {

            preprocess(args[0]);
            
        } else {
            System.out.println("<prog> <input file>");
        }
        
        System.err.println(new Timestamp(new Date().getTime()));
    }
        
        
}
