(ns ml.helpers)

(defn power
  [number pow]
  #?(:clj (Math/pow number pow)
     :cljs (js/Math.pow number pow)))

(defn sum
  [x]
  (apply + x))

(defn abs
  [x]
  #?(:clj (Math/abs x)
     :cljs (js/Math.abs xbv )))

(defn column
  [M j]
  (mapv #(get % j)
        M))

(defn M-minus
  [A B]
  (mapv (fn [A-row B-row]
          (mapv #(- %1 %2)
                A-row B-row))
        A B))

(defn M-element-multiply
  [x M]
  (mapv (fn [M-row]
          (mapv #(* x %) M-row))
        M))

(defn activation
  [x w]
  (sum (map (fn [x-element w-element]
              (* x-element w-element))
            x w)))

(defmulti activation-function
  (fn [activation-fn] activation-fn))

(defmethod activation-function :tanh
  [_]
  #?(:clj #(Math/tanh %)
     :cljs #(js/Math.tanh %)))

(defmethod activation-function :identity
  [_]
  identity)

(defmulti error-backpropagate
  (fn [activation-fn _] activation-fn))

(defmethod error-backpropagate :tanh
  [_ {:keys [hidden-unit output-errors w]}]
  (let [derivative (- 1 (power hidden-unit 2))
        summation (sum (map #(* %1 %2)
                            output-errors w))]
    (* derivative summation)))

(defmulti basis-function
  (fn [name _] name))

(defmethod basis-function :neural-network
  [_ {:keys [x w activation-fn]}]
  (-> x (activation w) ((activation-function activation-fn))))

(defmulti linear-model
  (fn [name _] name))

(defmethod linear-model :regression
  [_ {:keys [w basis-fns activation-fn]}]
  ((activation-function activation-fn) (sum (map #(* %1 %2)
                                                 w basis-fns))))