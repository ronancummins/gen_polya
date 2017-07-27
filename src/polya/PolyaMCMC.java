
package polya;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.math3.util.FastMath;

/**
 *
 * Estimate a generalised Polya urn model using Gradient Descent. 
 * 
 * 
 * @author rc635
 */
public class PolyaMCMC {
    
    public static class Entry{
        int cf;
        int df;
        double w;
        int idx;
        public Entry(int i){
            idx = i;
        }
    }
    
    private int[][] data;
    
    private TreeMap<String, Entry> dictionary;
    
    private int V;
    private int C;
    private int N;
    private int D;
    
    
    private String[] words;
    private double[] weights0;
    private double[] weights1;
    private double[] dynWeights;
    private double[] rep0;
    private double[] rep1;
    private double[] Eweights0;
    private double[] Erep0;
    private double[] mleWeightsMult;
    
    private double Enorm;
    private double normRep;
    private double normW;
    private double normDynW;
    private double loglike0;
    private double loglike1;
    
    private Random rand;
    
    private double var;
    private int iters= 500000;
    private int burnin=50000;
    
    public static int MULT = 0;
    public static int DCM = 1;
    public static int GenPLOYA = 2;
    public static int GenPLOYAcfdf = 3;
    
    public int distribtion = 0;
    
    private int accepted=0;
    private int rejected=0;
    private TreeMap<String, Entry> bg_dict =null;
    
    

    
    public PolyaMCMC(int[][] _data, String[] _words, int _dist, double _var, int _iters, int _burnin, TreeMap<String, Entry> _background_dict) throws FileNotFoundException, IOException{
        
        bg_dict = _background_dict;
        dictionary = new TreeMap();
        var = _var;
        distribtion = _dist;
        iters = _iters;
        burnin = _burnin;
        rand = new Random();
        
        
        words = _words;
        data = _data;

        C = 0;
        D = 0;
        N = 0;
        V = 0;
        
        
        for (int i=0;i<data.length;i++){
            HashSet<String> df = new HashSet();

            for (int j = 0; j < data[i].length; j++) {
                Entry e;
                
                if (dictionary.containsKey(words[data[i][j]])) {
                    e = dictionary.get(words[data[i][j]]);
                } else {
                    e = new Entry(V);
                    dictionary.put(words[data[i][j]], e);
                    V++;
                }
                
                e.cf++;
                

                if (!df.contains(words[data[i][j]])) {
                    e.df++;
                    df.add(words[data[i][j]]);
                }
            }
            D += df.size();
            N++;
            C+=data[i].length;
        }
        

        
        
        System.out.println("#docs: " + N + "\t #words: " + C + "\t dictionary: " + V + "\t #sum_doc_vecs: " + D );
        
        
        //weights are exponents of the polya parameters
        weights0 = new double[V];
        weights1 = new double[V];
        dynWeights = new double[V];
        rep0 = new double[V];
        rep1 = new double[V];
        
        //these hold the expectation of the accepted samples
        Eweights0 = new double[V]; 
        Erep0 = new double[V]; 
        
        //hold the max-likelihood of mult
        mleWeightsMult = new double[V]; 

            
        
        
        
        initUniformWeights();

        
        if (this.distribtion == PolyaMCMC.GenPLOYAcfdf) {
            if (bg_dict == null) {
                initHeuristicReplace(dictionary);
            } else {
                initHeuristicReplace(bg_dict);
            }
            this.distribtion = PolyaMCMC.DCM;
        }else if (this.distribtion == PolyaMCMC.GenPLOYA) {
            if (bg_dict == null) {
                initUniformReplace();
            }else{
                initSupervisedReplace(bg_dict);
                this.distribtion = PolyaMCMC.DCM;
            }
            
        }else {
            initUniformReplace();
        }
        
    }
    
    private static String[] removeNonWords(String[] sent){
        
        ArrayList<String> s = new ArrayList();

        for (int i=0;i<sent.length;i++){
            if (i>1){
                s.add(sent[i]);
            }
        }
        String[] ret = new String[s.size()];
        ret = s.toArray(ret);
        return ret;
    }
    
    /**
     * initialise weights to uniform 1s
     */
    private void initUniformReplace() {
        double n = 0;

        for (int i = 0; i < this.rep0.length; i++) {
            this.rep0[i] = 0.0;
            n += exp(this.rep0[i]);
        }
        this.normRep = n;

    }

    /**
     * initialise replacement weights to uniform log(cf/df)
     */
    private void initHeuristicReplace(TreeMap<String,Entry> bg_dictionary) {
        double n = 0;

        Entry e1, e2;
        for (String w:dictionary.keySet()){
            e1 = bg_dictionary.get(w);
            e2 = dictionary.get(w);
            this.rep0[e2.idx] = log((((double)e1.cf/(double)e1.df)));
            n += exp(this.rep0[e2.idx]);
        }
         //System.out.println(n);
        this.normRep = n;
    }

    
    /**
     * initialise replacement weights to learned ones
     */
    private void initSupervisedReplace(TreeMap<String,Entry> bg_dictionary) {
        double n = 0;

        Entry e1, e2;
        for (String w:dictionary.keySet()){
            e1 = bg_dictionary.get(w);
            e2 = dictionary.get(w);
            this.rep0[e2.idx] =  e1.w;
            n += exp(this.rep0[e2.idx]);
        }
         //System.out.println(n);
        this.normRep = n;
    }
    
    
    /**
     * initialise replacement weights from file
     */
    private static void initReplaceFromFile(TreeMap<String,Entry> bg_dictionary, String filename) throws FileNotFoundException, IOException {
        
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line, toks[];
            double n = 0;
            while ((line = br.readLine()) != null) {        
                toks = line.split("\t");
                Entry e = bg_dictionary.get(toks[0]);
                
                double w = Double.parseDouble(toks[6]);
                e.w = log(w);
            }
        }

    }    
    
    
    /**
     * initialise weights to uniform 1s
     */
    private void initUniformWeights() {
        double n = 0;
        for (int i = 0; i < this.weights0.length; i++) {
            this.weights0[i] = 0;
            n += exp(this.weights0[i]);
        }
        this.normW = n;
    }
    
   

    
    /**
     * multinomial-ml weights
     */
    private void setMLMulitnomialWeights() {
        double n = 0;
        Entry e;
        for (String w:dictionary.keySet()){
            e = dictionary.get(w);
            mleWeightsMult[e.idx] = log((double)e.cf);
            
        }

    }    
    

    
    private static double exp(double x){
        return Math.exp(x);
        //return FastMath.exp(x);
    }
    
    private static double log(double x){
        return Math.log(x);
        //return FastMath.log(x);
    }
    
    /**
     * return the loglikelihood of all the data
     * @param docs
     * @param w
     * @param norm
     * @return 
     */
    
    private double likelihood(int[][] docs, double[] w, double norm, double[] replace){
        
        double loglikelihood=0.0;
        for (int i=0;i<docs.length;i++){
            loglikelihood += this.docLoglikelihood(docs[i], w, norm,replace);
        }
        return loglikelihood;
    }
    
    /**
     * return the loglikelihood of one doc (sample)
     * @param doc
     * @param w
     * @param norm
     * @return 
     */
    private double docLoglikelihood(int[] doc, double[] w, double norm, double[] replace) {

        double doc_loglikelihood = 0.0;
        //copy the initial parameters  
        dynWeights = w.clone();
        normDynW = norm;
        int idx;
        for (int i=0;i<doc.length;i++){
            idx = doc[i];
            doc_loglikelihood += (dynWeights[idx]) - log(normDynW);
            
            //update the evolution of urn
            if (this.distribtion != PolyaMCMC.MULT){
                dynWeights[idx] = log(exp(dynWeights[idx]) + exp(replace[idx]));
                normDynW += exp(replace[idx]);
            }
        }
        
        return doc_loglikelihood;
    }
    
    
    /**
     * Generate the next sample from the current
     * and return the norm of the next sample
     * These are for the initial weights
     */
    public double nextCandidate(double[] next, double[] cur, boolean normalise, double new_norm) {
        double norm = 0.0;
        double noise;
        for (int i = 0; i < V; i++) {
            noise = rand.nextGaussian() * var;
            next[i] = cur[i] + noise;
            norm += exp(next[i]);
        }
        
        /**
        //normalise the sample
        if (normalise) {
            for (int i = 0; i < V; i++) {
                next[i] = log(new_norm*exp(next[i])/norm);
            }
            return new_norm;
        }else{
            return norm;
        }
        */
        return norm;
    }
    
    /**
     * create a new proposal using a gaussian 
     * from weights0 and place new proposal in weights1
     * calculates likelihood until acceptance
     */
    public void nextSample() {

        boolean accept = false;

        double normW1 = 0.0;
        double noise;

        //System.out.println((double)accepted/(double)(accepted+rejected));
        normW1 = nextCandidate(weights1, weights0, true, 1.0);
        if (this.distribtion == PolyaMCMC.GenPLOYA) {
            nextCandidate(rep1, rep0, false, 1.0);
            loglike1 = this.likelihood(this.data, weights1, normW1, rep1);
        }else{
            loglike1 = this.likelihood(this.data, weights1, normW1, rep0);
        }
        
        


        if (loglike1 > loglike0) {
            accept = true;
            accepted++;
        } else {
            double acc = rand.nextDouble();
            if (log(acc) < (loglike1 - loglike0)) {
                accept = true;
                accepted++;
            } else {
                rejected++;
            }
        }

        
        if (accept) {
            //accepted -- so swap samples (i.e. 1 to 0 etc)
            double swap[] = weights0;
            weights0 = weights1;
            weights1 = swap;
            normW = normW1;
            loglike0 = loglike1;

            if (this.distribtion == PolyaMCMC.GenPLOYA) {
                swap = rep0;
                rep0 = rep1;
                rep1 = swap;
            }
        }

    }
    
    
    /**
     * Run the mcmc
     */
    public void mcmc(){
        
        loglike0 = likelihood(this.data,weights0,normW,rep0);
        
        for (int i=0;i<iters;i++){
            
            this.nextSample();
            //System.out.println(i + "\t" + loglike0 + " -> " + loglike1);
            
            //after burn-in period, store parameters
            //for expectation
            if (i>=burnin){
                for (int j=0;j<V;j++){
                    Eweights0[j] += weights0[j];
                    Erep0[j] += rep0[j];
                    //System.out.println( rep0[j] + "\t" +  exp(rep0[j]));
                }
                //System.out.println("#####################");
            }
            
            if (((accepted+rejected)%5000 ==0)){
                System.out.println(new Timestamp(new Date().getTime()) + "\t" + (accepted+rejected) + " samples with acceptance rate: " + (double)accepted/(double)(accepted+rejected));
            }
        }
        
        Enorm = 0.0;
        for (int j = 0; j < V; j++) {
            Eweights0[j] = Eweights0[j]/(iters-burnin);
            Erep0[j] = Erep0[j]/(iters-burnin);
            Enorm += exp(Eweights0[j]);
        }
        
        
        System.out.println("Accepted: " + this.accepted + "\nRejected: "+ this.rejected + "\nSamples used (less burn-in):" + (iters-burnin) + "\nAcceptance rate: " + (double)accepted/(double)(accepted+rejected));
        double l = this.likelihood(data, Eweights0, Enorm, Erep0);
        System.out.println("likelihood of E \t" + l);
        System.out.println("Perplexity of E (nats) \t" + Math.exp(-l/(double)this.C));

        
    }
    
    /**
     * Print out the params and some statistics 
     */
    public void printOutput(double[] statistic){
            

        System.out.println("E[|w|]:\t" + Enorm);
        double n = 0.0;
        for (String w : this.dictionary.keySet()) {
            Entry item = dictionary.get(w);
            int idx = item.idx;

            System.err.println(w + "\t" + exp(statistic[idx]) + "\t" + exp(statistic[idx]) / Enorm + "\t" + item.cf + "\t" + item.df + "\t" + (double) item.cf / (double) item.df + "\t" + exp(Erep0[item.idx]));
        }

        System.err.println();
    }
    
    /**
     * sort and print out weights
     */
    public String toString(double[] w){
        StringBuilder s = new StringBuilder();
        double mass = 0.0;
        s.append("[");
        for (int i=0;i<w.length;i++){
            s.append(exp(w[i])).append(", ");
            mass += exp(w[i]);
            //s.append(w[i]).append(", ");
        }
        s.append("] = " + mass);
        return s.toString();
    }
        

    
    // return data and stats
    //
    
    public static int[][] read_data(String filename, ArrayList<String> word_list, TreeMap<String, Entry> dictionary) throws FileNotFoundException, IOException{
        
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        int lines = 0;
        while (reader.readLine() != null) lines++;
        reader.close();
        int N=0, C=0, V=0, D=0;
        int[][] data = new int[lines][];        
        
        
        //open a file an read documents in line by line (memory)
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line, words[];
            while ((line = br.readLine()) != null) {
                words = line.split("\\s+");
                
                words = removeNonWords(words);
                data[N] = new int[words.length];
                HashSet<String> df = new HashSet();
               
                for (int i=0;i<words.length;i++){
                    C++;
                    Entry e;
                    
                    if (dictionary.containsKey(words[i])){
                        e = dictionary.get(words[i]);
                    }else{
                        e = new Entry(V);
                        dictionary.put(words[i], e);
                        word_list.add(words[i]);
                        V++;
                    }
                    e.cf++;

                    data[N][i] = e.idx;
                    
                    if (!df.contains(words[i])){
                        e.df++;
                        df.add(words[i]);
                    }
                }
                D += df.size();
                N++;
                
            }
            
        }  
        
        
        return data;
        
    }
    

    public static int[][] data_instance(int[][] data, ArrayList<String> word_list, int i, ArrayList<String> word_list_instance){
        
        int V=0;
        TreeMap<String, Entry> dictionary = new TreeMap();
        int[][] dout = new int[1][data[i].length];
        
        Entry e;
        String w;
        for (int j=0;j<data[i].length;j++){
            w = word_list.get(data[i][j]);
            if (dictionary.containsKey(w)){
                e = dictionary.get(w);
            }else{
                e = new Entry(V);
                dictionary.put(w, e);
                word_list_instance.add(w);
                V++;
            }
            
            dout[0][j] = e.idx;
        }
        return dout;
    }
    
    
    public static void main(String[] args) throws IOException {
        
        
        System.out.println(new Timestamp(new Date().getTime()));
        if (args.length >= 6) {

            //load data
            ArrayList<String> word_list = new ArrayList();
            TreeMap<String, Entry> dictionary = new TreeMap();
            int[][] d = PolyaMCMC.read_data(args[0], word_list, dictionary);
            String[] arr_word_list = new String[word_list.size()];
            arr_word_list = word_list.toArray(arr_word_list);
            
            if (new Integer(args[5]) == 1){
                
                PolyaMCMC gp = new PolyaMCMC(d, arr_word_list, new Integer(args[1]), new Double(args[2]), new Integer(args[3]), new Integer(args[4]), null);

                System.out.println(new Timestamp(new Date().getTime()));

                gp.mcmc();
                gp.printOutput(gp.Eweights0);

                //print out the likelihood of multinomial mle model as 
                gp.setMLMulitnomialWeights();
                gp.distribtion = PolyaMCMC.MULT;
                double mle_l = gp.likelihood(gp.data, gp.mleWeightsMult, (double) gp.C, gp.Erep0);
                System.out.println("likelihood of ml mult \t" + mle_l);
                System.out.println("Perplexity of ml mult (nats) \t" + Math.exp(-mle_l / (double) gp.C));            
                
            }else{
                if (new Integer(args[1]) == PolyaMCMC.GenPLOYA){
                    if (args.length > 6){
                        PolyaMCMC.initReplaceFromFile(dictionary, args[6]);
                    }else{
                        System.out.println("<prog> <data-file> <dist (0=mult, 1=DCM, 2=GenPolya, 3=GenPolya (with bs_t set))> <variance> <# of samples> <burn-in period> <est_bg 1 or 0> <optional-background stats file when dist=2 and est_bg=0>");
                        System.exit(-1);
                    }
                }
                for (int i=0;i<d.length;i++){
                    
                    ArrayList<String> word_list_instance = new ArrayList();
                    int[][] din = PolyaMCMC.data_instance(d, word_list,i,word_list_instance);
                    String[] arr_word_list_inst = new String[word_list_instance.size()];
                    arr_word_list_inst = word_list_instance.toArray(arr_word_list_inst);
                    //System.out.println(word_list_instance.toString());
                    //System.out.println(Arrays.toString(din[0]));
                    //re set all initial stats for each doc
                    PolyaMCMC gp = new PolyaMCMC(din, arr_word_list_inst, new Integer(args[1]), new Double(args[2]),new Integer(args[3]),new Integer(args[4]),dictionary);
                    
                    System.out.println(new Timestamp(new Date().getTime()));

                    System.out.println("doc #" + i);
                    gp.mcmc();
                    gp.printOutput(gp.Eweights0);

                    //print out the likelihood of multinomial mle model as 
                    gp.setMLMulitnomialWeights();
                    gp.distribtion = PolyaMCMC.MULT;
                    double mle_l = gp.likelihood(gp.data, gp.mleWeightsMult, (double) gp.C, gp.Erep0);
                    System.out.println("likelihood of ml mult \t" + mle_l);
                    System.out.println("Perplexity of ml mult (nats) \t" + Math.exp(-mle_l / (double) gp.C));            
                }
                
            }
            

            
            
        } else {
            System.out.println("<prog> <data-file> <dist (0=mult, 1=DCM, 2=GenPolya, 3=GenPolya (with bs_t set))> <variance> <# of samples> <burn-in period> <est_bg 1 or 0> <optional-background stats file when dist=2 and est_bg=0>");
        }
        
        System.out.println(new Timestamp(new Date().getTime()));
    }
    


}
