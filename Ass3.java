import java.io.*;
import java.util.*;

public class Ass3 {
    public static void main(String[] args) {
        // ;oad the data from the fule
                List<Instance> patients = loadCSV("data.csv");

          // Shuffle the dataset randomly
                 // helps that we find the true accuracy of the model


                 for (int n=50 ; n<= 450 ; n+=100)
                  {
                    
                      for (int k=3 ; k<=7 ; k+=2)
                  {

                    
                      Collections.shuffle(patients);
                      Collections.shuffle(patients);
                      Collections.shuffle(patients);
                      System.out.println("The number of training index is: " + n);
                          System.out.println("The number of neighbors is: " + k);

                   // resepect ratio 4:1
                        int testSize = n / 4;


               // We select a training set , size N
                             List<Instance> TrainingList = patients.subList(0, n);

              // Build the k-d tree 
                            KdTreeClass KD_Tree = new KdTreeClass();


             //start timer
                              long startTime = System.nanoTime();


             //Build tree using only trainings set
                            KD_Tree.newTree(TrainingList);

             // we select a testing set , size testSize 
             // set has to be different from the first one , we cannot have intersetion of the sets
                          List<Instance> TestingList = patients.subList(n, n + testSize);
             //evaluate tree on testing set
                              testTree(KD_Tree, TestingList,k);

              // end timer and print it
                     System.out.println("Timing "  + ":"  + (System.nanoTime()-startTime)/1000000000.0 + " seconds");System.out.println();
             }
       }
        
     }



    // the tree has alreayd been built , we use the testing set and try predicitng the diagnosis
    // we calculate accuracy, and print number of correct and wrong predictions
    private static void testTree(KdTreeClass KD_Tree, List<Instance> TestingList , int k) {
        int correctCount = 0;
        int wrongCount = 0;

        for (Instance instance : TestingList) {

            // HOW MANY NEIHBORS 3 , 5 OR 7 ??????????newTreerecursive
            // will be decided later

                 List<Instance> Neighbors = KD_Tree.findNeighbors(instance.id,k, TestingList);
              String predictedDiagnosis = performVoting(Neighbors);
            
            // instance diagnosis is available in the csv file
            // predicted diagnosis is the one we get from the model
            // compare and update the counters

                      if (predictedDiagnosis.equals(instance.diagnosis)) {  correctCount++;
                      } else { wrongCount++;}
        }
        // normal printing counts and accuracy 
        System.out.println("Correct predictions: " + correctCount);
        System.out.println("Wrong predictions: " + wrongCount);
        double Accuracy = (double) correctCount / (correctCount + wrongCount)  * 100;
        System.out.println("Accuracy: " + Accuracy + "%");

    }





    // The following method loads the data from the CSV file into an arraylist
    // it uses BufferedRader to read the file and RegEx to split the data

    private static List<Instance> loadCSV(String filename) {
        List<Instance> patients = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
               String line;
                  boolean isFirstLine = true;
                     while ((line = reader.readLine()) != null) {
                           if (isFirstLine) {isFirstLine = false;
                          continue; // Skip the first line               
                               }

                   String[] index = line.split(",");
                    String id = index[0];
                 String diagnosis = index[1];
                     double[] attributes = new double[10];
                     for (int i = 0; i < 10; i++) {
                      attributes[i] = Double.parseDouble(index[i + 2]);
                }
                Instance instance = new Instance(id, diagnosis, attributes);
                patients.add(instance);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return patients;
    }



    // The following method implements a hashmap to oerform voting
    // it originates from the following resource : https://stackoverflow.com/questions/33717514/voting-program-using-hashmap-with-java 

    private static String performVoting(List<Instance> patients) {
          Map<String, Integer> votes = new HashMap<>();
                for (Instance instance : patients) {
                  votes.put(instance.diagnosis, votes.getOrDefault(instance.diagnosis, 0) + 1);
        }

           String maxVoteDiagnosis = null;
               int maxVoteCount = 0;

        for (Map.Entry<String, Integer> entry : votes.entrySet()) {
                if (maxVoteDiagnosis == null || entry.getValue() > maxVoteCount) {
                    maxVoteDiagnosis = entry.getKey();
                    maxVoteCount = entry.getValue();
            }
        }

        return maxVoteDiagnosis;
    }
}





class Instance {
    String id,diagnosis;
    
    double[] attributes;

    public Instance(String id, String diagnosis, double[] attributes) {
        this.id = id;
        this.diagnosis = diagnosis;
        this.attributes = attributes;
    }
}

// Define a class to represent a node in the k-d tree or ball tree
class Node {
    Instance instance;
    Node left,right;

    public Node(Instance instance) {
        this.instance = instance;
        this.left = null;
        this.right = null;
    }
}




class KdTreeClass {
    Node root;

    public void newTree(List<Instance> patients) {
        root = BuildTree(patients, 0);
    }

    private Node BuildTree(List<Instance> patients, int depth) {
        if (patients.isEmpty())
            return null;

        int axis = depth % patients.get(0).attributes.length;
        patients.sort(Comparator.comparingDouble(inst -> inst.attributes[axis]));

        int medianIndex = patients.size() / 2;
        Node node = new Node(patients.get(medianIndex));

        node.left = BuildTree(patients.subList(0, medianIndex), depth + 1);
        node.right = BuildTree(patients.subList(medianIndex + 1, patients.size()), depth + 1);

        return node;
    }




    // using the distance between tqwo points , we find the closest neighboars and store them in a list
    // k will be defined in the main program
    //InstanceReference will be explained in its own method

    public List<Instance> findNeighbors(String id, int k, List<Instance> patients) {
        InstanceReference InstanceReference = new InstanceReference(null);
        for (Instance instance : patients) {
            if (instance.id.equals(id)) {
                InstanceReference.targetInstance = instance;
                break;
            }
        }

        if (InstanceReference.targetInstance == null) {
            System.out.println("Instance with ID " + id + " not found.");
            return null;
        }

        PriorityQueue<Instance> kNeighbors = new PriorityQueue<>(k, (inst1, inst2) -> {
            double dist1 = calculateDistance(inst1.attributes, InstanceReference.targetInstance.attributes);
            double dist2 = calculateDistance(inst2.attributes, InstanceReference.targetInstance.attributes);
            return Double.compare(dist2, dist1);
        });

        search(root, InstanceReference, k, kNeighbors);

        List<Instance> Neighbors = new ArrayList<>();
        while (!kNeighbors.isEmpty()) {
            Neighbors.add(kNeighbors.poll());
        }

        return Neighbors;
    }

    private void search(Node node, InstanceReference InstanceReference, int k, PriorityQueue<Instance> Neighbors) {
        if (node == null)
            return;

        double distance = calculateDistance(node.instance.attributes, InstanceReference.targetInstance.attributes);
        if (Neighbors.size() < k || distance < calculateDistance(Neighbors.peek().attributes, InstanceReference.targetInstance.attributes)) {
            Neighbors.offer(node.instance);
            if (Neighbors.size() > k)
                Neighbors.poll();
        }

        int axis = Neighbors.size() % InstanceReference.targetInstance.attributes.length;
        double axisDistance = InstanceReference.targetInstance.attributes[axis] - node.instance.attributes[axis];

        if (axisDistance < 0) {
            search(node.left, InstanceReference, k, Neighbors);
            if (Neighbors.size() < k || Math.abs(axisDistance) < calculateDistance(Neighbors.peek().attributes, InstanceReference.targetInstance.attributes))
                search(node.right, InstanceReference, k, Neighbors);
        } else {
            search(node.right, InstanceReference, k, Neighbors);
            if (Neighbors.size() < k || Math.abs(axisDistance) < calculateDistance(Neighbors.peek().attributes, InstanceReference.targetInstance.attributes))
                search(node.left, InstanceReference, k, Neighbors);
        }
    }



    // The following method calculates the distance between two patients
    // it uses sqrt (a^2 + b^2 + c^2 ... + j^2) to calculate the distance between all 10 attributes
    private double calculateDistance(double[] attributes1, double[] attributes2) {
        double distanceSquared = 0.0;
        for (int i = 0; i < attributes1.length; i++) {
            double diff = attributes1[i] - attributes2[i];
            distanceSquared += diff * diff;
        }
        return Math.sqrt(distanceSquared);
    }
}


// after debugging problem there was problem with instance being modified and being nullified during process

class InstanceReference {
    Instance targetInstance;

    public InstanceReference(Instance targetInstance) {
        this.targetInstance = targetInstance;
    }
}