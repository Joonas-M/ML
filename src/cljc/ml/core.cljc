(ns ml.core
  (:require [ml.helpers :as helpers]))

(defn map-vecs->vec-maps [m]
  (let [foo (seq m)
        m-keys (map first foo)
        m-vals (map second foo)]
    (apply map (fn [& args]
                 (zipmap m-keys args))
           m-vals)))

(defn ready-function
  [w-first-layer w-second-layer]
  (fn [x]
    (let [x (conj x 1)
          hidden-units (map #(helpers/basis-function :neural-network {:x x :w % :activation-fn :tanh})
                            w-first-layer)]
      (helpers/linear-model :regression {:w (first w-second-layer) :basis-fns hidden-units :activation-fn :identity}))))

(defmulti neural-network
          (fn [_ {method :method}]
            method))

(defmethod neural-network :feed-forward
  [observations
   {:keys [max-iterations learning-rate gradient-limit] :as options}]
  (let [vector-valued-observation? (vector? (:x (first observations)))
        max-iterations (or max-iterations 100)
        M 4
        learning-rate (or learning-rate 0.4)
        gradient-limit (or gradient-limit 0.005)]
    (loop [weights []]
      (when-not (empty? weights)
        (let [w-first-layer (first weights)
              w-second-layer (second weights)
              ready-function (ready-function w-first-layer w-second-layer)
              y-diff (mapv #(- (first (:t %)) (ready-function (:x %))) observations)]
          (println "READY ----> " (pr-str weights))
          (println "SUM -------> " (pr-str (apply + y-diff)))
          (println "--------------------------")))
      (recur (reduce (fn [layer-weights {:keys [x t] :as observation}]
                       (let [x (conj x 1)]
                         (loop [i 1
                                w-first-layer (if (empty? layer-weights)
                                                (into [] (repeat M
                                                                 (-> (count x) (- 1) (take (iterate inc 1)) vec (conj 1))))
                                                (first layer-weights))
                                w-second-layer (if (empty? layer-weights)
                                                 (into [] (repeat (count t)
                                                                  (-> M (- 1) (take (iterate inc 1)) vec (conj 1))))
                                                 (second layer-weights))]
                           (if (= i 2)
                             [w-first-layer w-second-layer]
                             (let [hidden-units (map #(helpers/basis-function :neural-network {:x x :w % :activation-fn :tanh})
                                                     w-first-layer)
                                   y (map #(helpers/linear-model :regression {:w % :basis-fns hidden-units :activation-fn :identity})
                                          w-second-layer)
                                   output-errors (map #(- %1 %2) y t)
                                   hidden-unit-errors (map-indexed (fn [index hidden-unit]
                                                                     (helpers/error-backpropagate :tanh {:hidden-unit hidden-unit
                                                                                                         :output-errors output-errors
                                                                                                         :w (helpers/column w-second-layer index)}))
                                                                   hidden-units)
                                   first-layer-error-derivatives (mapv (fn [error]
                                                                         (mapv #(* error %)
                                                                               x))
                                                                       hidden-unit-errors)
                                   second-layer-error-derivatives (mapv (fn [error]
                                                                          (mapv #(* error %)
                                                                                hidden-units))
                                                                        output-errors)]
                               (recur (inc i)
                                      (helpers/M-minus w-first-layer (helpers/M-element-multiply learning-rate first-layer-error-derivatives))
                                      (helpers/M-minus w-second-layer (helpers/M-element-multiply learning-rate second-layer-error-derivatives))))))))
                     weights observations)))))

;;;;;; DECISION TREE ;;;;;;;

(defn information-gain
  []
  ;;TODO
  )

(declare best-attribute)

(defn gini
  [explained-data [data-key explaining-data]]
  (let [classes (distinct explained-data)
        groups (distinct explaining-data)
        groups-n (reduce (fn [grouping data-point]
                           (update grouping
                                   data-point
                                   (fn [group-count]
                                     (if group-count
                                       (inc group-count) 1))))
                         {} explaining-data)
        foo (map (fn [x y]
                   [x y])
                 explaining-data explained-data)
        bar (group-by identity foo)
        asd (reduce (fn [result [[group-bar class] pairs]]
                      (update result group-bar #(conj % (/ (count pairs) (get groups-n group-bar)))))
                    {} bar)
        gini-index (fn [class-propotions]
                     (apply - 1 (map #(* % %) class-propotions)))]
    (apply + (map (fn [group-bar]
                    (* (/ (get groups-n group-bar)
                          (apply + (vals groups-n)))
                       (gini-index (get asd group-bar))))
                  groups))))

(defmulti best-attribute
          (fn [explained-data explaining-data-sets options]
            (:type (meta explained-data))))

(defmethod best-attribute :categorical
  [explained-data explaining-data-sets {:keys [cost-function] :as options}]
  (let [cost-fn (case cost-function
                  (:gini nil) gini
                  :information-gain information-gain
                  (throw (#?(:clj  Exception.
                             :cljs js/Error.) (str "Can't use " cost-function " as a cost function for categorical explained data. "
                                                   "Valid values are :gini and :information-gain. If left undefined, :gini is used."))))
        costs (map (fn [[data-key explaining-data :as foo-bar]]
                     (let [find-split-point? (= :numerical (-> explaining-data meta :type))
                           split-point-data (when find-split-point?
                                              (let [sorted-values (sort explaining-data)
                                                    mid-values (:values (reduce (fn [{:keys [values previous-value]} value]
                                                                                  {:values (conj values
                                                                                                 (+ previous-value (/ (- value previous-value) 2)))
                                                                                   :previous-value value})
                                                                                {:values []
                                                                                 :previous-value (first sorted-values)}
                                                                                (rest sorted-values)))]
                                                (best-attribute explained-data
                                                                (into {}
                                                                      (map (fn [value]
                                                                             [value (with-meta (map #(> value %) explaining-data)
                                                                                               {:type :categorical})]))
                                                                      mid-values)
                                                                :gini)))]
                       {:cost (cost-fn explained-data [data-key (or (-> split-point-data :data vals first) explaining-data)])
                        :data (into {} [foo-bar])
                        :split-point (-> split-point-data :data keys first)}))
                   explaining-data-sets)
        sorted-costs (sort-by :cost costs)]
    (case cost-function
      (:gini nil) (first sorted-costs)
      :information-gain (last sorted-costs))))

(defmethod best-attribute :numerical
  [explained-data explaining-data-sets options]
  (let [split-value (fn [ys-l ys-r]
                      (let [s-l (apply + ys-l)
                            s-r (apply + ys-r)
                            n-l (count ys-l)
                            n-r (count ys-r)]
                        (+ (/ (* s-l s-l)
                              n-l)
                           (/ (* s-r s-r)
                              n-r))))
        possible-split-points (map (fn [[data-key explaining-data :as foo-bar]]
                                     (case (-> foo-bar meta :type)
                                       :numerical (let [foo (map-vecs->vec-maps {:x explaining-data
                                                                                 :y explained-data})
                                                        bar (sort-by :x foo)]
                                                    (reduce (fn [{:keys [i best-split-value previous-split-point] :as split}
                                                                 {split-point :x y :y}]
                                                              (let [ys-l (take i (map :y bar))
                                                                    ys-r (drop i (map :y bar))
                                                                    new-split-value (split-value ys-l ys-r)]
                                                                (if (> new-split-value best-split-value)
                                                                  {:i (inc i)
                                                                   :best-split-value new-split-value
                                                                   :split-point (/ (+ split-point previous-split-point)
                                                                                   2)
                                                                   :previous-split-point split-point}
                                                                  (assoc split
                                                                    :i (inc i)
                                                                    :previous-split-point split-point))))
                                                            {:i 0 :best-split-value 0
                                                             :previous-split-point 0} bar))
                                       :categorical "FOO"   ;;TODO
                                       ))
                                   explaining-data-sets)]
    (last (sort-by :best-split-value possible-split-points))))

(defn most-common-value
  [values]
  (key (apply max-key #(-> % val count) (group-by identity values))))

(defn attribute-branches
  [[data-key explaining-data]]
  (case (:type (meta explaining-data))
    :numerical [true false]
    :categorical (distinct explaining-data)))

(defn attribute-test
  [{:keys [split-point data]}]
  (fn [data-point]
    (if split-point
      (->> data keys first (get data-point) (> split-point))
      (->> data keys first (get data-point)))))

(defn form-branch-data
  [data explained-variable test branch]
  (let [values-to-keep (map #(= branch (test %)) (map-vecs->vec-maps data))]
    (with-meta
      (reduce-kv (fn [m k v]
                   (assoc m k (with-meta (vec (keep identity
                                                    (map #(when %2 %1) v values-to-keep)))
                                         (meta v))))
                 {} data)
      (meta data))))

(defn train-decision-tree
  [data {:keys [stop-criterion] :as options}]
  ;; Data on muotoa ^{:y :jotain}{:data-sarakkeen-nimi ^{:type :categorical}["punainen" "keltainen"]}
  (let [explained-variable (:y (meta data))
        explained-data (explained-variable data)
        explaining-data-sets (dissoc data explained-variable)]
    (if (or (<= (count explained-data) stop-criterion)
            (= 1 (count (into #{} explained-data))))
      {:leaf (most-common-value explained-data)}
      (let [attribute-data (best-attribute explained-data explaining-data-sets options)
            branches (attribute-branches (first (:data attribute-data)))
            test (attribute-test attribute-data)
            node (apply merge
                        {:test test
                         :attribute-data (update attribute-data :data #(-> % keys first))
                         :n-data-points (or (:n-new-data-points options) (-> data vals first count))}
                        (map (fn [branch]
                               (let [new-data (form-branch-data data explained-variable test branch)
                                     n-new-data-points (-> new-data vals first count)
                                     n-data-points (-> data vals first count)]
                                 (cond
                                   (= 0 n-new-data-points) {:leaf nil
                                                            :n-data-points n-new-data-points}
                                   (= n-data-points n-new-data-points) {:leaf (first explained-data)
                                                                        :n-data-points n-new-data-points}
                                   :else {branch (train-decision-tree (form-branch-data data explained-variable test branch) (assoc options
                                                                                                                               :n-data-points n-new-data-points))})))
                             branches))]
        (if (and (contains? node :test)
                 (contains? node :leaf))
          (select-keys node [:leaf])
          node)))))

(defn apply-tree [tree data]
  (if-let [test (:test tree)]
    (apply-tree (get tree (test data)) data)
    (:leaf tree)))