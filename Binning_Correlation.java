

import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Scanner;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;

class MyComparator implements Comparator {

	Map map;
	
	public MyComparator(Map map) {
	    this.map = map;
	}

	public int compare(Object o1, Object o2) {
	
	    return ((Double) map.get(o2)).compareTo((Double) map.get(o1));
	
	}
}


public class DP {
	
	public static int rows = 0;
	public static int cols = 0;
	public static int NumIntervals = 3; // intervals
    public static int K = 200; // value of K
    public static int NumCorrelatedPairs=100; // number of top 100 correltaed pairs 

	public static void main(String[] args) throws IOException {

		
		
		
		Scanner s2 = new Scanner(new BufferedReader(new FileReader(args[0])));


		
	     while (s2.hasNextLine())
	     {
	         String d = s2.nextLine();
	         StringTokenizer st = new StringTokenizer(d,",");
	         cols = st.countTokens();

	         rows ++;
	         
	     }

		double [][] genes = new double [rows][cols]; 
		
		s2.close();
		
		Scanner s1 = new Scanner(new BufferedReader(new FileReader(args[0])));

	    int r=0;    
	    String tokens;
	    String positive = "positive";
	    String negative = "negative";
	    while (s1.hasNextLine())
	     {
	         String d = s1.nextLine();
	         StringTokenizer st = new StringTokenizer(d,",");

	         int c = 0;
	         while(st.hasMoreTokens()) {
	        	 tokens = st.nextToken();
	         	if(tokens.equals(positive))
	         		genes[r][c] = 1;
	         	else if(tokens.equals(negative))
	         		genes[r][c] = 0;
	         	else
	         	genes[r][c] = Double.parseDouble(tokens);

	        	 
	             //genes[r][c] = Double.parseDouble(st.nextToken());
	             c++;
	         }
	         r++;
	     }
	     
               
        //entropy calculation
        entropy_based(genes);
        //correlation calculation
        correlation_coefficient(genes);
	
	
        FileWriter fileWriter = null;
        FileWriter fileWriter1 = null;


		fileWriter = new FileWriter("edensitybins.csv");
		fileWriter1 = new FileWriter("edensitydata.csv");


		
		double [] gene = new double[rows];
        int counting = 0;
        int NumIntervals = 3;
        int elements = rows/NumIntervals; 
        int element_count = elements;
        double [] variance_bin = new double[elements];

        int range = NumIntervals;
        int remaining = rows%NumIntervals;
        int [] bin_count = new int[elements];
        int[] bin_index = new int[elements];
    	double mean =0;


        int bin_i, i, var;
        double variance_cal =0, variance, var_inter=0;

	    for( int col = 0; col< K; col++)

            {
	    	bin_i=0;
        	elements = element_count;
        	range = NumIntervals;
        	remaining = rows%NumIntervals;

           	            
            for(int j=0; j<rows; j++)
            {
            	gene[j] = genes[j][col];
            	
            }
            	
            java.util.Arrays.sort(gene);
            
            counting =0;
            
            for( i =0; i<(rows-element_count); i++)
            {
            	if(counting<(element_count-1)) {
            		counting++;
            	}
            	else {
            		if(gene[i] == gene[i +1]){
            			bin_count[bin_i] = element_count + 1;

            			bin_index[bin_i] = i;
            			range--;
            			remaining--;
            			bin_i++;
            			counting = 0;
            			i = i+1;
            		}
            		
            		else {
            			if(range == remaining) {
            				if(gene[i+1] == gene[i+2])
            				{
            					remaining = remaining-2;
            					bin_index[bin_i] = i+2;
            					bin_count[bin_i] = element_count +2;
            					counting = 0;
            					i = i +2;
            					bin_i++;
            					range--; 

            				}
            				else {
            					counting = 0;
            					bin_count[bin_i] = element_count +1;
            					bin_index[bin_i] = i +1;
            					bin_i++;
            					remaining--;
            					range--;

            				}
            			}
            			else {
            				counting = 0;
            				bin_index[bin_i] = i;
            				bin_count[bin_i] = element_count;
            				range--;
            				bin_i++;

            			}
            		}
            	
            	}
            }
            	
            	if( range == 1 && remaining == 1 ) {
            		bin_count[bin_i] = element_count + 1;
            		bin_index[bin_i] = rows-1;

            	}
            	else {
            		bin_count[bin_i] = element_count;
            		bin_index[bin_i] = rows-1;
            	}
            	            	
            	for(i =0; i<NumIntervals; i++)
            	{
            		bin_i = bin_index[i];
            		
            		if (i ==0)
            		{
            			for(int s = 0; s<bin_i; s++ ) 
            				mean = mean + gene[s];
            			
            			for(int s =0; s<bin_i; s++)
            				variance_cal = (gene[s] - mean) * (gene[s] - mean);
            			
            			variance_bin[i] = variance_cal/bin_count[i];
            		}
            		else {
            			
            			var = bin_index[i-1];
            			for(int s = var; s<bin_i; s++ ) 
	            			mean = mean + gene[s];
            			
            			for(int s =var; s<bin_i; s++)
            				variance_cal = (gene[s] - mean) * (gene[s] - mean);
            			
            			variance_bin[i] = variance_cal/bin_count[i];
            			
            		}
            		
            	}
                       for(int m = 0; m<NumIntervals; m++) 
            			{
                    	   var_inter += variance_bin[m];
            			}
            			
            			variance = var_inter/element_count;	  
            			
            			
            			printing(gene, bin_count, bin_index, fileWriter, variance, col, fileWriter1);
            			
            }
	    fileWriter.close();
	    fileWriter1.close();

	}
	

	    public static void printing(double [] gene, int [] bin_count, int[] bin_index, FileWriter fileWriter, double variance, int col, FileWriter fileWriter1 ) throws IOException
	    {

			int bin_i, var;
            			
            			
	                     fileWriter.append("g" + (col+1));
	                     fileWriter.append(" Variance:  ");
	                     fileWriter.append(String.valueOf(variance));
	                     
	                     double initalvalue = Double.NEGATIVE_INFINITY;
			               double finalvalue = Double.POSITIVE_INFINITY;
	           				byte edata = 97;
  
		            			
		            		for(int i = 0; i<NumIntervals; i++)
		            		{
		            			bin_i = bin_index[i];
		            			
		            			
			            		
			            		if (i ==0)
			            		{
			                      fileWriter.append("(" + String.valueOf(initalvalue) + "; " + String.valueOf((int)Math.round(((gene[bin_i]+gene[bin_i+1])/2))) + ")"  );
				                  fileWriter.append(",");
				                  fileWriter.append(String.valueOf(bin_count[i]));
				                  fileWriter.append(",");
				                  
				                  for(int ind = 0; ind<=bin_i; ind++)
				            		{
				                	  fileWriter1.append((char)edata + ";  ");	
				            		}

			            		}
			            		
			            		else if(i == (NumIntervals-1))
			            		{
			            			var = bin_index[i-1];

				                    fileWriter.append("(" + String.valueOf((int)(Math.round((gene[var]+gene[var+1])/2))) + "; " + String.valueOf(finalvalue) + ")"  );
					                fileWriter.append(",");
					                fileWriter.append(String.valueOf(bin_count[i]));
			            		}
			            		
			            		else {
			            			var = bin_index[i-1];

				                    fileWriter.append("[" + String.valueOf((int)(Math.round((gene[var]+gene[var+1])/2))) + "; " + String.valueOf((int)(Math.round((gene[var]+gene[var+1])/2))) + ")"  );
					                fileWriter.append(",");
					                fileWriter.append(String.valueOf(bin_count[i]));
					                  fileWriter.append(",");
			            		}
			            		
			            		if(i >= 1)
			            		{
			            			var = bin_index[i-1];
			            			edata++;
					                  for(int ind=(var+1); ind<=bin_i; ind++ ){
					                	  fileWriter1.append((char)edata + ";  ");	

					                  }
			            		}

			            		
			            		
		            		}
			                fileWriter.append("\n");
			                fileWriter1.append("\n");


		          }
	                     
	 

public static double logs(double num)
{
	return (Math.log(num)/Math.log(2));
}
	
public static void correlation_coefficient(double [][] genes) throws IOException {
		

		double [] gene1 = new double[rows];
        double [] gene2 = new double[rows];
        double [] dy = new double[rows];
        double [] dxdy = new double[rows];
        double g1mean=0.0, g2mean = 0.0, cov_sum =0.0, sd1 = 0.0;
        double sd2 = 0.0, sx = 0, sy =0, covariance;
        int n, u =0;
        double [] dx = new double[rows];

        Map <String, Double> geneMap= new HashMap<String, Double>();    
        n = ((cols-1)*(cols-2))/2;

        
        for(int k =0; k<(cols-2); k++)
        {
        	for(int j=k+1; j<(cols-1);j++)
        	{
        		for(int i=0;i<rows;i++)
        		{
        			gene1[i] = genes[i][k];
        			g1mean = g1mean + gene1[i];
        			gene2[i] = genes[i][j];
        			g2mean = g2mean + gene2[i];

        		}
        		
        		g1mean = g1mean/rows;
        		g2mean = g2mean/rows;
        		
        		for(int l=0; l<rows; l++)
        		{
        			dx[l] = gene1[l] -g1mean;
        			dy[l] = gene2[l] -g2mean;
        			dxdy[l] = dx[l] * dy[l];
        			cov_sum = dxdy[l] + cov_sum;
        			sd1 = (dx[l]*dx[l]) + sd1;
        			sd2 = (dy[l]*dy[l]) + sd2;
       			
        		}
        		
        		covariance = cov_sum/(rows-1);
        		sx = Math.sqrt(sd1/(rows-1));
        		sy = Math.sqrt(sd2/(rows-1));
 
        		geneMap.put("G"+(k+1)+" G"+(j+1), covariance/(sx * sy));
        	}
        }
       
        MyComparator comp=new MyComparator(geneMap);

        Map<String, Double> newMap = new TreeMap(comp);
        newMap.putAll(geneMap);

        printMap(newMap);

	}


public static void printMap(Map mp) throws IOException {
    FileWriter file = null;
	  file = new FileWriter("correlatedgenes.csv");
    Iterator it = mp.entrySet().iterator();
    
    int i =0; 
    
    while (it.hasNext()) {
    	if(i<NumCorrelatedPairs)
    	{
        Map.Entry pair = (Map.Entry)it.next();
        file.append( String.valueOf(pair.getValue()));
        file.append(",");
        file.append( String.valueOf(pair.getKey()));
        file.append("\n");
        it.remove(); 
    	}
    	else
    		break;	
        i++;
        }
    file.close();
}


public static void printMaping(Map mp, double[] finalmidvalue,  double [][] genes) throws IOException {
    FileWriter file = null;
	  file = new FileWriter("entropybins.csv");
    Iterator it = mp.entrySet().iterator();
    int count =0;
    while (it.hasNext()) {
    	
    	if(count <K) {
    		
        Map.Entry pair = (Map.Entry)it.next();
       
        file.append( String.valueOf(pair.getKey()));
        file.append("\n");
        it.remove();     
        }
    	else 
    		break;
    	count++;
        }
    file.close();
    
    
    file = new FileWriter("entropydata.csv");
    
	for (int i = 0; i< rows; i++)
	{
		for(int j =0; j<(cols-1); j++)
		{
			if (genes[i][j] <=  finalmidvalue[j])
			{
		        file.append("a; ");
			}
			if (genes[i][j] >  finalmidvalue[j])
			{
				file.append("b; ");
			}

		}
		file.append(", ");
		file.append(String.valueOf(genes[i][cols-1])+"\n");
	}

    file.close();
    
    
}



	public static void entropy_based(double[][] genes) throws IOException {
		

		Double [][] sortedlist = new Double [(cols-1)][2];
		
        Map <String, Double> geneMap= new HashMap<String, Double>();    


        int v =0, length=0;
        int classes = (cols-1);

        double [] entropygain = new double[rows];
        double pos = 0, neg=0, total2,  total, total1;
        int k =0;
        double entropy=0, entropy1=0, entropy2=0, gain1=0;
        double [] spliting = new double[rows-1];
        
        double maximum = Double.NEGATIVE_INFINITY;

        double [][] gene = new double [rows][2];

        double [] finalmidvalue = new double[cols-1];

        
        int u =-1;
        
        for( int col = 0; col< (cols-1); col++)

        {

        u = -1;
        maximum = Double.NEGATIVE_INFINITY;

        
        for(int j=0; j<rows; j++)
        {
        	gene[j][v] = genes[j][col];
        	gene[j][1] = genes[j][classes];
        	
        }
        	
        java.util.Arrays.sort(gene, new java.util.Comparator<double[]>() {
            public int compare(double[] a, double[] b) {
                return Double.compare(a[0], b[0]);
            }
        });
        
        
                
        for (k =0; k<rows; k++)
        {
        	if(gene[k][1]==0)
        		neg += 1;
        	if(gene[k][1] == 1)
        		pos += 1;
        }
        
        total = pos + neg;
        
        entropy = -(((pos/total)*logs(pos/total)) + ((neg/total)*logs(neg/total)));
        
        for(length=0; length<(rows-1); length++)
        {
        	
        	int split = 0;
	        double pos1=0, neg1=0,pos2=0, neg2=0;
	       
	        spliting[length] = (gene[length][v] + gene[length+1][v])/2;
	        
	        for (k =0; k<rows; k++)	
	        {
	        	if(gene[k][v] <= spliting[length])
	        		split = split + 1;
	        }
                
	       for(k =0;k<split; k++)
	       {
	    	   if(gene[k][1]==0)
	       		neg1 += 1;
	       		if(gene[k][1] == 1)
	       		pos1 += 1;
	       }
	       
	       for(k =split;k<rows; k++)
	       {
	    	   	if(gene[k][1]==0)
	       		neg2 += 1;
	       		if(gene[k][1] == 1)
	       		pos2 += 1;
	       }
	
       
	       total1 = pos1 + neg1;
	       
	       total2 = pos2 + neg2;

	             
	       entropy1 = -(((pos1/total1)*logs(pos1/total1)) + ((neg1/total1)*logs(neg1/total1)));
	       entropy2 = -(((pos2/total2)*logs(pos2/total2)) + ((neg2/total2)*logs(neg2/total2)));

	       
	       if (Double.isNaN(entropy1))
	    	   entropy1 = 0;

	      
	       if (Double.isNaN(entropy2))
	    	   entropy2 = 0;
		
	       gain1 = ((total1/(total1+total2))*entropy1) +  ((total2/(total1+total2))*entropy2);	
	       entropygain[length] = entropy - gain1;
        
        }
        	
        for (int h =0; h<(rows-1); h++)
        {

        	if(entropygain[h] > maximum)
        	{
        		maximum = entropygain[h];
        		u = h;
        	}
        		
        }
        
		geneMap.put("g" + (col+1) + ":   Info gain: " + (maximum) +   "; Bins (-; "+ spliting[u] + "]  " +(u+1)+"   ("+(spliting[u] + "; +]   " + (rows-(u+1))), maximum);
		
        finalmidvalue[col] = spliting[u];                      

        }
        
        MyComparator comp=new MyComparator(geneMap);

        Map<String, Double> newMap = new TreeMap(comp);
        newMap.putAll(geneMap);
        
        printMaping(newMap, finalmidvalue, genes);
		}
	
	
}

