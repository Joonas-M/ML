(ns ml.core
  (:require [ml.helpers :as helpers]))

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
#_(defn gini
    [{:keys [groups total-data-count]}]
    (let [gini-index-fn #(- 1 (apply + (map (fn [p] (* p p)) %)))]
      (apply + (map (fn [[x-category y]]
                      (let [y-vals-count (apply + (vals y))]
                        (* (gini-index-fn (map #(/ % y-vals-count) (vals y)))
                           (/ y-vals-count total-data-count))))
                    groups))))

#_(defn group-cost-fn
    [x-y-categories]
    (reduce (fn [result {:keys [x-category y-category] :as bar}]
              (update result x-category (fn [ys-mapped]
                                          (update ys-mapped y-category (fn [y-count] (if y-count (inc y-count) 1))))))
            {} x-y-categories))

#_(defn start-cost-fn
    [x-category-values explained-data cost-fn]
    (let [foo (group-cost-fn (map (fn [x y]
                                    {:x-category x
                                     :y-category y})
                                  x-category-values explained-data))
          explained-data-count (count explained-data)]
      {:cost-value (cost-fn {:groups foo
                             :total-data-count explained-data-count})
       :decision (map (fn [[input-value responses]]
                        [input-value (last (sort-by val responses))])
                      foo)}))

#_(defn find-split-point
    [values explained-data explained-variable-type cost-function]
    (let [cost-fn (case cost-function
                    :gini gini
                    :information-gain information-gain)
          reduce-fn (if (= :categorical explained-variable-type)
                      (fn [result value]
                        (let [x-category (map #(>= value %) values)
                              foo (group-cost-fn (map (fn [x y]
                                                        {:x-category x
                                                         :y-category y})
                                                      x-category explained-data))
                              explained-data-count (count explained-data)
                              cost-value (cost-fn {:groups foo
                                                   :total-data-count explained-data-count})]
                          (case cost-function
                            :gini (if (< (:cost result) cost-value) result {:cost cost-value :split-point value})))))]
      (reduce reduce-fn
              {:cost 1 :split-point nil} values)))

#_(defn create-node
    [data numerical-data-fns explained-data cost-fn]
    (first
      (sort-by :cost-value
               (map (fn [[data-header row-data]]
                      (assoc (if (contains? numerical-data-fns data-header)
                               (start-cost-fn (map #((data-header numerical-data-fns) %) row-data) explained-data cost-fn)
                               (start-cost-fn row-data explained-data cost-fn))
                        :decision-fn (if (contains? numerical-data-fns data-header)
                                       (data-header numerical-data-fns)
                                       identity)
                        :header data-header))
                    data))))

(declare best-attribute)

(defn gini
  [explained-data [data-key explaining-data] {print? :print?}]
  (when print? (println "-->EXPLAINED DATA"))
  (when print? (clojure.pprint/pprint explained-data))
  (when print? (println "-->EXPLAINING DATA"))
  (when print? (clojure.pprint/pprint explaining-data))
  (let [classes (distinct explained-data)
        groups (distinct explaining-data)
        groups-n (reduce (fn [grouping data-point]
                           (update grouping
                                   data-point
                                   (fn [group-count]
                                     (if group-count
                                       (inc group-count) 1))))
                         {} explaining-data)
        _ (when print? (println "GROUPS N"))
        _ (when print? (clojure.pprint/pprint groups-n))
        foo (map (fn [x y]
                   [x y])
                 explaining-data explained-data)
        _ (when print? (println "FOO"))
        _ (when print? (clojure.pprint/pprint foo))
        bar (group-by identity foo)
        _ (when print? (println "BAR"))
        _ (when print? (clojure.pprint/pprint bar))
        asd (reduce (fn [result [[group-bar class] pairs]]
                      (update result group-bar #(conj % (/ (count pairs) (get groups-n group-bar)))))
                    {} bar)
        _ (when print? (println "ASD"))
        _ (when print? (clojure.pprint/pprint asd))
        gini-index (fn [class-propotions]
                     (apply - 1 (map #(* % %) class-propotions)))]
    (apply + (map (fn [group-bar]
                    (when print?
                      (println "GINI INDEX VALUES")
                      (clojure.pprint/pprint (gini-index (get asd group-bar))))
                    (* (/ (get groups-n group-bar)
                          (apply + (vals groups-n)))
                       (gini-index (get asd group-bar))))
                  groups))))

(defmulti best-attribute
          (fn [explained-data explaining-data-sets cost-function foo-opt]
            (:type (meta explained-data))))
(defmethod best-attribute :categorical
  [explained-data explaining-data-sets cost-function foo-opt]
  (let [cost-fn (case cost-function
                  (:gini nil) gini
                  :information-gain information-gain
                  (throw (#?(:clj  Exception.
                             :cljs js/Error.) (str "Can't use " cost-function " as a cost function for categorical explained data. "
                                                   "Valid values are :gini and :information-gain. If left undefined, :gini is used."))))
        costs (map (fn [[data-key explaining-data :as foo-bar]]
                     (let [find-split-point? (= :numerical (-> explaining-data meta :type))
                           split-point-data (when find-split-point?
                                              (best-attribute explained-data
                                                              (into {}
                                                                    (map (fn [value]
                                                                           [value (with-meta (map #(> value %) explaining-data)
                                                                                             {:type :categorical})]))
                                                                    explaining-data)
                                                              :gini {:print? false}))]
                       (when split-point-data
                         (println "SPLIT POINT DATA")
                         (clojure.pprint/pprint split-point-data))
                       {:cost (cost-fn explained-data [data-key (or (-> split-point-data :data vals first) explaining-data)] foo-opt)
                        :data (into {} [foo-bar])
                        :split-point (-> split-point-data :data keys first)}))
                   explaining-data-sets)]
    #_(println "COSTS")
    #_(clojure.pprint/pprint costs)
    (case cost-function
      (:gini nil) (->> costs (sort-by :cost) first)
      :information-gain (->> costs (sort-by :cost) last))))
(defmethod best-attribute :numerical
  [explained-data explaining-data cost-function foo-opt]
  ;;TODO Pitäs käyttää regressiota
  )

(defn most-common-value
  [values]
  #_(println "MOST COOMNS: " (pr-str values))
  (key (apply max-key #(-> % val count) (group-by identity values))))

(defn attribute-branches
  [[data-key explaining-data]]
  #_(println "EXPLAINEIN DA: " (pr-str explaining-data))
  (case (:type (meta explaining-data))
    :numerical [true false]
    :categorical (distinct explaining-data)))

(defn attribute-test
  [{:keys [split-point data]}]
  (fn [data-point]
    (if split-point
      (->> data keys first (get data-point) (> split-point))
      (->> data keys first (get data-point)))))

(defn map-vecs->vec-maps [m]
  (let [foo (seq m)
        m-keys (map first foo)
        m-vals (map second foo)]
    (apply map (fn [& args]
                 (zipmap m-keys args))
           m-vals)))

(defn form-branch-data
  [data explained-variable test branch]
  (let [values-to-keep (map #(= branch (test %)) (map-vecs->vec-maps data))]
    #_(println "VALUES TO KEEP:" (pr-str values-to-keep))
    (with-meta
      (reduce-kv (fn [m k v]
                   (assoc m k (with-meta (vec (keep identity
                                                      (map #(when %2 %1) v values-to-keep)))
                                         (meta v))))
                 {} data)
      (meta data))))

(defn train-decision-tree
  [data {:keys [stop-criterion cost-function] :as options}]
  ;; Data on muotoa ^{:y :jotain}{:data-sarakkeen-nimi ^{:type :categorical}["punainen" "keltainen"]}
  (let [explained-variable (:y (meta data))
        cost-fn (case cost-function
                  :gini gini
                  :information-gain information-gain)
        explained-data (explained-variable data)
        explaining-data-sets (dissoc data explained-variable)
        #_#_split-points-for-numerical-data (into {} (comp
                                                       (filter #(= :numerical (:type (meta (second %)))))
                                                       (map (fn [[name values]]
                                                              [name (find-split-point values
                                                                                      explained-data
                                                                                      explained-variable-type
                                                                                      cost-function)])))
                                                  (dissoc data explained-variable))
        #_#_numerical-data-fns (into {} (map (fn [[data-header split-result]]
                                               {data-header #(>= % (:split-point split-result))})
                                             split-points-for-numerical-data))]
    (if (<= (count explained-data) stop-criterion)
      {:leaf (most-common-value explained-data)}
      (let [attribute-data (best-attribute explained-data explaining-data-sets cost-function {:print? false})
            #_#__ (println "ATTRIBUTE DATA")
            #_#__ (clojure.pprint/pprint attribute-data)
            branches (attribute-branches (first (:data attribute-data)))
            test (attribute-test attribute-data)]
        #_(println "BRANCHES: " (pr-str branches))
        (apply merge
               {:test test}
               (map (fn [branch]
                      (let [new-data (form-branch-data data explained-variable test branch)
                            #_#__ (println "DATA: ")
                            #_#__ (clojure.pprint/pprint data)
                            #_#__ (println "NEW DATA")
                            #_#__ (clojure.pprint/pprint new-data)
                            n-new-data-points (-> new-data vals first count)
                            n-data-points (-> data vals first count)]
                        (cond
                          (= 0 n-new-data-points) {:leaf nil}
                          (= n-data-points n-new-data-points) {:leaf (first explained-data)}
                          :else {branch (train-decision-tree (form-branch-data data explained-variable test branch) options)})))
                    branches))))
    #_(loop [decision-functions []
             explained-variables #{explained-variable}]
        (if stop-criterion
          decision-functions
          (let [node (create-node (apply dissoc data explained-variables) numerical-data-fns explained-data cost-fn)]
            (recur (conj decision-functions #(((:decision-fn node) %) (:decision node)))
                   (conj explained-variables (:header node))))))))