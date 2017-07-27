package polya;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;



/**
 *
 * @author ronanc
 */
public class Retrieval {
    
    public static class Tuple implements Comparable<Tuple>{
        String key;
        double param;
         
         
        public int compareTo(Tuple t) {
            if (this.param > t.param){
                return -1;
            }else if (this.param < t.param){
                return 1;
            }else {
                return 0;
            }
             
        }
         
        public String toString(){
            return key + " " + param;
        }
    }
    
    
    TreeMap<String, String[]> docs;
    TreeMap<String, String[]> qrys;
    TreeMap<String, Set<String>> qrels;
    
    TreeMap<String, Double> bg_model;
    TreeMap<String, TreeMap<String, Double>> doc_models;
    
    int N;  // # docs
    int C;  // # tokens
    int V;  // # vocab
    int D;  // sum of dfs
    
    double mass; 
    
    int mult = 0;
    
    public Retrieval(String docs_fname, String qrys_fname, String doc_stats_fname, String bg_stats_fname, String qrels_fname, String _mult) throws IOException{
        
        mass = 200;
        mult = Integer.parseInt(_mult);
        this.index(docs_fname, docs = new TreeMap());
        this.index(qrys_fname, qrys = new TreeMap());
        
        this.load_stats(doc_stats_fname, doc_models = new TreeMap());
        TreeMap<String, TreeMap<String, Double>> temp_models;
        C = this.load_stats(bg_stats_fname, temp_models = new TreeMap());
        bg_model = temp_models.firstEntry().getValue();

        //System.out.println(this.docs.size() + "\t" + this.doc_models.size());
        qrels = new TreeMap();
        this.load_qrels(qrels_fname);

    }
    
    
    
    private void load_qrels(String qrels_fname) throws FileNotFoundException, IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(qrels_fname))) {
            String line, parts[];
            while ((line = br.readLine()) != null) {

                parts = line.split("\t");
                //System.out.println(Arrays.toString(parts));
                if (!qrels.containsKey(parts[0])) {
                    qrels.put(parts[0].trim(), new TreeSet());
                }
                //System.out.println("-" + parts[0] + "-" + parts[1] + "-");
                qrels.get(parts[0].trim()).add(parts[1]);

            }
        }

    }
    
    // indexes the queries and documents
    private void index(String filename, Map<String, String[]> m) throws FileNotFoundException, IOException{
        
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line, parts[], id;
            
            while ((line = br.readLine()) != null) {
                parts = line.split("\t");
                id = parts[0].split(" ")[1];
                //System.out.println(parts.length);
                if (parts.length > 1){
                    m.put(id.trim(), parts[1].split(" "));
                }
            }
        }
    }
    
    // loads the preprocessed statistics for background and docs
    private int load_stats(String filename, TreeMap<String, TreeMap<String, Double>> m) throws FileNotFoundException, IOException{
        int sum=0;
        int i=0;
        TreeMap<String, Double> cur = new TreeMap();
        m.put(String.valueOf(i+1), cur);
        
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line, parts[], id;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()){
                    i++;
                    cur = new TreeMap();
                    m.put(String.valueOf(i+1), cur);
                }else{
                    
                    parts = line.split("\t");
                    if (mult == 1){
                        cur.put(parts[0].trim(), Double.parseDouble(parts[3]));
                        sum += cur.get(parts[0].trim());
                    }else{
                        cur.put(parts[0].trim(), Double.parseDouble(parts[2]));
                    }
                }
                
            }
        }
        return sum;
    }
    
    
    
    
    public double ap(String qid, Tuple[] res){
        double ap=0;
        double f=0;

        for (int i=0;i<1000;i++){
            //System.out.print(qid +"-" +res[i].key + "-" + res[i].param);
            if (this.qrels.get(qid).contains(res[i].key)){
               f++; 
               ap += f/(i+1); 
               //System.out.print("****");
            }
            //System.out.print("\n");
        }
        //System.out.println(qid + "\t" + ap/this.qrels.get(qid).size() + "\t" + this.qrels.get(qid).size());
        return ap/this.qrels.get(qid).size();
        
    }
    
    
    public void run_all_queries(){
        Tuple[] res;
        double avp, map=0.0;
        
        for (String qid:this.qrys.keySet()){
            if (qrels.containsKey(qid)){
                res = this.run_query(this.qrys.get(qid));
                Arrays.sort(res);
                avp = ap(qid, res);
                map+=avp;
                //System.out.println(avp);
            }else{
                //System.out.println("skipping qry ... " + qid);
            }
        }
        
        map = map/this.qrels.size();
        System.out.println(map + "\t" + this.qrels.size());
    }
    
    
    
    public Tuple[] run_query(String[] Q){
        
        TreeMap<String, Double> doc;
        String[] doc_str;
        double loglike;
        
        Tuple[] res = new Tuple[doc_models.size()];
        int i=0;
        double dl;
        for (String id: this.doc_models.keySet()){
            
            doc = this.doc_models.get(id);
            doc_str = this.docs.get(id);
            if (doc_str == null){
                dl = 0;
            }else{
                dl = this.docs.get(id).length;
            }

            
            loglike=0.0;
            for (String q : Q){
                if (!bg_model.containsKey(q)) continue;
                if (doc.containsKey(q)){
                    if (mult == 0){
                        loglike += Math.log((doc.size()*doc.get(q) + this.mass*bg_model.get(q))/(doc.size() + this.mass));
                    }else{
                        if (mult == 1){
                            loglike += Math.log((doc.get(q) + this.mass*bg_model.get(q)/C)/(dl + this.mass));
                        }else{
                            loglike += Math.log((dl*doc.get(q) + this.mass*bg_model.get(q))/(dl + this.mass));
                        }
                    }
                }else{
                    if (mult == 0){
                        loglike += Math.log((this.mass*bg_model.get(q))/(doc.size() + this.mass));
                    }else{
                        if (mult == 1){
                            loglike += Math.log((this.mass*bg_model.get(q)/C)/(dl + this.mass));
                        }else{
                            loglike += Math.log((this.mass*bg_model.get(q))/(dl + this.mass));
                        }
                    }
                }
            }
            //System.out.println(loglike);
            res[i] = new Tuple();
            res[i].key = id;
            res[i].param = loglike;
            
            i++;
        }
        
        return res;
    }
    
    
    
    
    
    //main
    public static void main(String args[]) throws IOException{
        
        if (args.length != 6){
            System.out.println("prog (docs) (qrys) (doc_stats) (bg_stats) (qrels) (0=no_mult, 1=mle_mult, 2=mcmc_mult)");
            System.exit(-1);
        }
        
        
        Retrieval engine = new Retrieval(args[0],args[1],args[2],args[3], args[4], args[5]);
        
        //System.err.println(engine.doc_models.size() + "\t" + engine.docs.size());
        
        engine.mass = 10;
        engine.run_all_queries();

        engine.mass = 50;
        engine.run_all_queries();        
        
        engine.mass = 100;
        engine.run_all_queries();

        engine.mass = 200;
        engine.run_all_queries();

        engine.mass = 300;
        engine.run_all_queries();

        engine.mass = 400;
        engine.run_all_queries();

        engine.mass = 500;
        engine.run_all_queries();
        
        engine.mass = 600;
        engine.run_all_queries();

        engine.mass = 1000;
        engine.run_all_queries();        

        engine.mass = 10000;
        engine.run_all_queries();        
        
      
    }
    
}
