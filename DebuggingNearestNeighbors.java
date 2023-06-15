import java.io.*;
import java.util.*;

// Define a class to represent an instance with attributes
class Instance {
    String id;
    String diagnosis;
    double[] attributes;

    public Instance(String id, String diagnosis, double[] attributes) {
        this.id = id;
        this.diagnosis = diagnosis;
        this.attributes = attributes;
    }
}

// Define a class to represent a node in the tree
class Node {
    Instance instance;
    Node left;
    Node right;

    public Node(Instance instance) {
        this.instance = instance;
        this.left = null;
        this.right = null;
    }
}

// Define a mutable wrapper class to store the reference of target instance
class TargetInstanceWrapper {
    Instance targetInstance;

    public TargetInstanceWrapper(Instance targetInstance) {
        this.targetInstance = targetInstance;
    }
}

// Define the main class for kNN classification
class KNNClassifier {
    Node root;

    public void buildTree(List<Instance> instances) {
        root = buildTreeRecursive(instances, 0);
    }

    private Node buildTreeRecursive(List<Instance> instances, int depth) {
        if (instances.isEmpty())
            return null;

        int axis = depth % instances.get(0).attributes.length;
        instances.sort(Comparator.comparingDouble(inst -> inst.attributes[axis]));

        int medianIndex = instances.size() / 2;
        Node node = new Node(instances.get(medianIndex));

        node.left = buildTreeRecursive(instances.subList(0, medianIndex), depth + 1);
        node.right = buildTreeRecursive(instances.subList(medianIndex + 1, instances.size()), depth + 1);

        return node;
    }

    public List<Instance> findNearestNeighbors(String id, int k, List<Instance> instances) {
        TargetInstanceWrapper targetInstanceWrapper = new TargetInstanceWrapper(null);
        for (Instance instance : instances) {
            if (instance.id.equals(id)) {
                targetInstanceWrapper.targetInstance = instance;
                break;
            }
        }

        if (targetInstanceWrapper.targetInstance == null) {
            System.out.println("Instance with ID " + id + " not found.");
            return null;
        }

        PriorityQueue<Instance> kNearestNeighbors = new PriorityQueue<>(k, (inst1, inst2) -> {
            double dist1 = calculateDistance(inst1.attributes, targetInstanceWrapper.targetInstance.attributes);
            double dist2 = calculateDistance(inst2.attributes, targetInstanceWrapper.targetInstance.attributes);
            return Double.compare(dist2, dist1);
        });

        searchKNearestNeighbors(root, targetInstanceWrapper, k, kNearestNeighbors);

        List<Instance> nearestNeighbors = new ArrayList<>();
        while (!kNearestNeighbors.isEmpty()) {
            nearestNeighbors.add(kNearestNeighbors.poll());
        }

        return nearestNeighbors;
    }

    private void searchKNearestNeighbors(Node node, TargetInstanceWrapper targetInstanceWrapper, int k, PriorityQueue<Instance> nearestNeighbors) {
        if (node == null)
            return;

        double distance = calculateDistance(node.instance.attributes, targetInstanceWrapper.targetInstance.attributes);
        if (nearestNeighbors.size() < k || distance < calculateDistance(nearestNeighbors.peek().attributes, targetInstanceWrapper.targetInstance.attributes)) {
            nearestNeighbors.offer(node.instance);
            if (nearestNeighbors.size() > k)
                nearestNeighbors.poll();
        }

        int axis = nearestNeighbors.size() % targetInstanceWrapper.targetInstance.attributes.length;
        double axisDistance = targetInstanceWrapper.targetInstance.attributes[axis] - node.instance.attributes[axis];

        if (axisDistance < 0) {
            searchKNearestNeighbors(node.left, targetInstanceWrapper, k, nearestNeighbors);
            if (nearestNeighbors.size() < k || Math.abs(axisDistance) < calculateDistance(nearestNeighbors.peek().attributes, targetInstanceWrapper.targetInstance.attributes))
                searchKNearestNeighbors(node.right, targetInstanceWrapper, k, nearestNeighbors);
        } else {
            searchKNearestNeighbors(node.right, targetInstanceWrapper, k, nearestNeighbors);
            if (nearestNeighbors.size() < k || Math.abs(axisDistance) < calculateDistance(nearestNeighbors.peek().attributes, targetInstanceWrapper.targetInstance.attributes))
                searchKNearestNeighbors(node.left, targetInstanceWrapper, k, nearestNeighbors);
        }
    }

    private double calculateDistance(double[] attributes1, double[] attributes2) {
        double distanceSquared = 0.0;
        for (int i = 0; i < attributes1.length; i++) {
            double diff = attributes1[i] - attributes2[i];
            distanceSquared += diff * diff;
        }
        return Math.sqrt(distanceSquared);
    }
}

public class DebuggingNearestNeighbors {
    //loop infinite times

    public static void main(String[] args) {
            while (true)
            {
        // Load the dataset from the CSV file
        List<Instance> instances = loadInstancesFromCSV("data.csv");

        // Build the k-d tree or ball tree
        KNNClassifier knn = new KNNClassifier();
        knn.buildTree(instances);

        // Prompt for the ID
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the ID to find nearest neighbors: ");
        String id = scanner.nextLine();

        // Inout the value of k
        Scanner scanner2 = new Scanner(System.in);
        System.out.print("Enter the value of k: ");
        int k = scanner2.nextInt();



        //int k = 5; // Choose a value of k
        List<Instance> nearestNeighbors = knn.findNearestNeighbors(id, k, instances);
        if (nearestNeighbors != null) {
            System.out.println("Nearest neighbors to ID " + id + ":");
            for (Instance neighbor : nearestNeighbors) {
                System.out.println("ID: " + neighbor.id + ", Diagnosis: " + neighbor.diagnosis);
            }

            // Perform voting and compare predicted diagnosis with actual diagnosis
            String predictedDiagnosis = performVoting(nearestNeighbors);
            String actualDiagnosis = getActualDiagnosis(id, instances);
            System.out.println("Predicted Diagnosis: " + predictedDiagnosis);
            System.out.println("Actual Diagnosis: " + actualDiagnosis);
        }
    }
    }

    private static List<Instance> loadInstancesFromCSV(String filePath) {
        List<Instance> instances = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean isFirstLine = true;
            while ((line = reader.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue; // Skip the header row
                }

                String[] values = line.split(",");
                String id = values[0];
                String diagnosis = values[1];
                double[] attributes = new double[10];
                for (int i = 0; i < 10; i++) {
                    attributes[i] = Double.parseDouble(values[i + 2]);
                }
                Instance instance = new Instance(id, diagnosis, attributes);
                instances.add(instance);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return instances;
    }

    private static String performVoting(List<Instance> nearestNeighbors) {
        Map<String, Integer> diagnosisCount = new HashMap<>();
        for (Instance neighbor : nearestNeighbors) {
            String diagnosis = neighbor.diagnosis;
            diagnosisCount.put(diagnosis, diagnosisCount.getOrDefault(diagnosis, 0) + 1);
        }

        String predictedDiagnosis = "";
        int maxCount = 0;
        for (Map.Entry<String, Integer> entry : diagnosisCount.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                predictedDiagnosis = entry.getKey();
            }
        }

        return predictedDiagnosis;
    }

    private static String getActualDiagnosis(String id, List<Instance> instances) {
        for (Instance instance : instances) {
            if (instance.id.equals(id)) {
                return instance.diagnosis;
            }
        }
        return "";
    }
}